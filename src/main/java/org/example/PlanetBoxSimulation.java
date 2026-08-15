package org.example;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.util.List;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.IntStream;

/**
 * PlanetBoxSimulation
 *
 * A 2D Newtonian gravity simulation of planets confined to an inescapable square
 * box. The box starts empty: press on the canvas where a body should appear, drag
 * towards the direction it should travel, and release to launch it. Its mass,
 * radius and launch speed come from the spawn sliders.
 * Planets attract each other via Newton's law of universal gravitation
 * (F = G * m1 * m2 / r^2) and bounce elastically (or inelastically, depending
 * on the restitution setting) off the walls of the box, like pool balls off a
 * cushion. Optionally, planets can also collide with each other.
 *
 * All physical parameters are adjustable at runtime via the control panel:
 *   - Gravitational constant G
 *   - Time step (dt)
 *   - Simulation speed (steps per frame)
 *   - Wall restitution: the fraction of a body's speed that survives each bounce
 *     off a cushion (1.0 = perfectly elastic, nothing lost)
 *   - Planet-planet collisions on/off, with their own restitution
 *   - Zero-radius mode: bodies have no contact cross-section, so they pass straight
 *     through one another; they are still drawn full size and still hit the walls
 *   - Gravity softening (prevents singular forces at tiny separations)
 *   - Mass, radius and launch speed given to the next body the user drags out
 *   - Trails on/off and trail length
 *   - "Show gravitational well": swaps the top-down view for a 3D one, where the
 *     (x,y) plane is drawn as a tilted rubber sheet whose z displacement is the
 *     local Newtonian potential, and each planet is a sphere resting on it
 *
 * Integration uses semi-implicit (symplectic) Euler, which conserves energy
 * far better than plain Euler for orbital mechanics.
 *
 * Compile:  javac PlanetBoxSimulation.java
 * Run:      java PlanetBoxSimulation
 */
public class PlanetBoxSimulation {

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Planets in a Box — Newtonian 2D Simulation");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            SimulationConfig config = new SimulationConfig();
            SimulationPanel simPanel = new SimulationPanel(config);
            ControlPanel controls = new ControlPanel(config, simPanel);

            frame.setLayout(new BorderLayout());
            frame.add(simPanel, BorderLayout.CENTER);
            frame.add(new JScrollPane(controls,
                    JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                    JScrollPane.HORIZONTAL_SCROLLBAR_NEVER), BorderLayout.EAST);

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            simPanel.start();
        });
    }

    // ------------------------------------------------------------------
    // Configuration shared between the physics engine and the UI.
    // Volatile fields so slider changes on the EDT are seen by the sim loop.
    // ------------------------------------------------------------------
    static class SimulationConfig {
        volatile double gravitationalConstant = 2000.0; // scaled G for pixel units
        volatile double timeStep = 0.005;               // dt per physics step
        volatile int stepsPerFrame = 4;                 // physics steps per rendered frame
        volatile double wallRestitution = 0.9;          // speed kept per wall bounce (1 = elastic)
        volatile double planetRestitution = 0.9;        // for planet-planet impacts
        volatile boolean planetCollisions = true;
        volatile boolean pointMasses = false;           // zero contact radius: bodies never collide
        volatile double softening = 4.0;                // gravity softening length (px)
        volatile boolean showTrails = true;
        volatile int trailLength = 200;
        volatile boolean showVectors = false;
        volatile boolean paused = false;

        // Gravitational well (3D space-time sheet) view:
        volatile boolean showGravityWell = true;
        volatile double wellDepthScale = 3.5;   // px of dip per unit of potential
        volatile double wellPitch = 58;         // camera tilt, degrees (0 = top-down)
        volatile double wellYaw = 22;           // camera rotation about the vertical, degrees
        volatile int wellResolution = 64;       // mesh cells across the box

        // Properties given to the next body the user launches:
        volatile double spawnMass = 400;
        volatile double spawnRadius = 12;
        volatile double spawnSpeed = 60;
        volatile boolean spawnFixed = false;    // pin the next body where it is placed
    }

    // ------------------------------------------------------------------
    // A single planet.
    // ------------------------------------------------------------------
    static class Planet {
        double x, y;        // position (px)
        double vx, vy;      // velocity (px/s)
        double ax, ay;      // acceleration accumulator
        double mass;
        double radius;
        Color color;
        /** Pinned in place: pulls on everything, is moved by nothing. */
        final boolean fixed;
        final Deque<Point2D> trail = new ArrayDeque<>();
        /** Scratch slot: where the renderer parked this body's prepared trail. */
        int renderIndex;

        Planet(double x, double y, double vx, double vy, double mass, double radius,
               Color color, boolean fixed) {
            this.x = x; this.y = y;
            this.vx = vx; this.vy = vy;
            this.mass = mass; this.radius = radius;
            this.color = color;
            this.fixed = fixed;
        }

        /**
         * Inverse mass, as the collision solver sees it. A fixed body behaves as if
         * infinitely massive, so impulses and overlap corrections leave it untouched
         * and the body that hit it takes the whole of both.
         */
        double invMass() { return fixed ? 0 : 1.0 / mass; }

        record Point2D(double x, double y) {}
    }

    // ------------------------------------------------------------------
    // The simulation canvas: physics loop + rendering.
    // ------------------------------------------------------------------
    static class SimulationPanel extends JPanel {

        // The heavy per-frame work — gravity, sampling the space-time sheet, projecting
        // and shading the mesh, projecting the trails — is split across the common
        // ForkJoinPool, whose workers plus the calling thread cover every core. Each
        // kernel is written so a worker only ever writes to the slice it owns, so none
        // of them need locks. Java2D itself is single-threaded, so the draw calls stay
        // on the caller; the kernels exist to leave that pass with as little to do as
        // possible.

        /**
         * Below these sizes a parallel dispatch costs more than it saves — measured
         * at roughly 30us of fan-out overhead on this pool — so the same kernels run
         * inline on the caller instead. The three differ because the work per element
         * differs: a gravity pair is a handful of flops, a sheet sample is a sqrt, and
         * a mesh cell also allocates its two Colors.
         */
        private static final int PARALLEL_BODY_THRESHOLD = 192;      // bodies (work ~ n^2)
        private static final long PARALLEL_SAMPLE_THRESHOLD = 20_000; // grid points x bodies
        private static final int PARALLEL_CELL_THRESHOLD = 1_024;     // mesh cells

        private final SimulationConfig cfg;
        private final List<Planet> planets = new CopyOnWriteArrayList<>();
        private final Random rng = new Random();
        private javax.swing.Timer timer;

        // Drag-to-launch state, in simulation-plane coordinates.
        private boolean dragging;
        private double dragFromX, dragFromY, dragToX, dragToY;

        SimulationPanel(SimulationConfig cfg) {
            this.cfg = cfg;
            setPreferredSize(new Dimension(820, 820));
            setBackground(new Color(8, 10, 24));

            // Press where the body should appear, drag towards where it should head,
            // release to launch it. The box starts empty; every body comes from here.
            MouseAdapter launcher = new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) {
                    double[] w = toPlane(e.getX(), e.getY());
                    dragFromX = w[0]; dragFromY = w[1];
                    dragToX = w[0];   dragToY = w[1];
                    dragging = true;
                    repaint();
                }
                @Override public void mouseDragged(MouseEvent e) {
                    if (!dragging) return;
                    double[] w = toPlane(e.getX(), e.getY());
                    dragToX = w[0]; dragToY = w[1];
                    repaint();
                }
                @Override public void mouseReleased(MouseEvent e) {
                    if (!dragging) return;
                    double[] w = toPlane(e.getX(), e.getY());
                    dragToX = w[0]; dragToY = w[1];
                    dragging = false;
                    planets.add(newBody(dragFromX, dragFromY, dragToX - dragFromX, dragToY - dragFromY));
                    repaint();
                }
            };
            addMouseListener(launcher);
            addMouseMotionListener(launcher);
        }

        void start() {
            timer = new javax.swing.Timer(16, e -> {  // ~60 FPS
                if (!cfg.paused) {
                    for (int i = 0; i < cfg.stepsPerFrame; i++) step(cfg.timeStep);
                }
                repaint();
            });
            timer.start();
        }

        void clearPlanets() {
            planets.clear();
        }

        void clearTrails() {
            for (Planet p : planets) p.trail.clear();
        }

        // ----------------- The square simulation plane -----------------
        //
        // The world is a box x box square centred in the panel, so the sheet in the
        // 3D view is square whatever shape the window happens to be.

        int boxSize()    { return Math.min(getWidth(), getHeight()); }
        int boxOriginX() { return (getWidth() - boxSize()) / 2; }
        int boxOriginY() { return (getHeight() - boxSize()) / 2; }

        /** Screen point -> simulation-plane coordinates, through the current view. */
        private double[] toPlane(double screenX, double screenY) {
            if (cfg.showGravityWell) return new WellView().unproject(screenX, screenY);
            return new double[]{screenX - boxOriginX(), screenY - boxOriginY()};
        }

        /**
         * A body launched from (x,y) heading along (dirX, dirY). The drag only sets
         * the direction — mass, radius and speed all come from the spawn sliders — so
         * a drag of any length gives the same launch speed. A fixed body ignores the
         * drag entirely and starts (and stays) at rest.
         */
        private Planet newBody(double x, double y, double dirX, double dirY) {
            double r = cfg.spawnRadius, box = boxSize();
            x = Math.max(r, Math.min(box - r, x));
            y = Math.max(r, Math.min(box - r, y));
            double[] v = launchVelocity(dirX, dirY);
            Color color = Color.getHSBColor(rng.nextFloat(), 0.65f + rng.nextFloat() * 0.35f, 0.95f);
            return new Planet(x, y, v[0], v[1], cfg.spawnMass, r, color, cfg.spawnFixed);
        }

        /** Launch velocity for a drag of (dirX, dirY): direction from the drag, speed from the slider. */
        private double[] launchVelocity(double dirX, double dirY) {
            double len = Math.hypot(dirX, dirY);
            if (cfg.spawnFixed || len <= 1e-6) return new double[]{0, 0};
            return new double[]{dirX / len * cfg.spawnSpeed, dirY / len * cfg.spawnSpeed};
        }

        /** Velocity the in-progress drag would launch with. */
        private double[] dragVelocity() {
            return launchVelocity(dragToX - dragFromX, dragToY - dragFromY);
        }

        // ----------------- Physics -----------------
        private void step(double dt) {
            int box = boxSize();
            if (box <= 0) return;

            Planet[] list = planets.toArray(new Planet[0]);

            // 1) Pairwise Newtonian gravity with Plummer softening.
            computeGravity(list);

            // 2) Semi-implicit Euler: update velocity first, then position.
            //    Fixed bodies are pinned: they still curve space for everyone else,
            //    but nothing they feel is allowed to move them.
            for (Planet p : list) {
                if (p.fixed) continue;
                p.vx += p.ax * dt;
                p.vy += p.ay * dt;
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }

            // 3) Planet-planet collisions (impulse-based, like billiard balls).
            //    Point masses have no cross-section, so there is nothing to resolve.
            if (cfg.planetCollisions && !cfg.pointMasses) {
                for (int i = 0; i < list.length; i++) {
                    Planet a = list[i];
                    for (int j = i + 1; j < list.length; j++) {
                        resolveCollision(a, list[j], cfg.planetRestitution);
                    }
                }
            }

            // 4) Inescapable box: reflect off the cushions. Restitution scales the
            //    whole velocity rather than just the reflected component, so every
            //    bounce costs speed — even a glancing one along the cushion.
            for (Planet p : list) {
                if (p.fixed) continue;
                double r = p.radius;
                boolean bounced = false;
                if (p.x - r < 0)        { p.x = r;         p.vx = Math.abs(p.vx);  bounced = true; }
                else if (p.x + r > box) { p.x = box - r;   p.vx = -Math.abs(p.vx); bounced = true; }
                if (p.y - r < 0)        { p.y = r;         p.vy = Math.abs(p.vy);  bounced = true; }
                else if (p.y + r > box) { p.y = box - r;   p.vy = -Math.abs(p.vy); bounced = true; }
                if (bounced) {
                    p.vx *= cfg.wallRestitution;
                    p.vy *= cfg.wallRestitution;
                }
            }

            // 5) Trails.
            if (cfg.showTrails) {
                for (Planet p : list) {
                    if (p.fixed) continue;   // a pinned body's trail is just a dot
                    p.trail.addLast(new Planet.Point2D(p.x, p.y));
                    while (p.trail.size() > cfg.trailLength) p.trail.removeFirst();
                }
            }
        }

        /**
         * Accelerations from pairwise Newtonian gravity.
         *
         * Each body sums the pull of every other body into its own accumulator, so no
         * two threads ever write to the same object and the whole O(n^2) sweep splits
         * cleanly across cores. That costs twice the pair evaluations of the symmetric
         * "compute once, apply to both" form, but the symmetric form writes to two
         * bodies per pair and cannot be parallelised without locking or per-thread
         * accumulator buffers — above a few dozen bodies, cores win easily.
         *
         * Positions are snapshotted into flat arrays first: the inner loop then walks
         * contiguous memory instead of chasing object references.
         */
        private void computeGravity(Planet[] ps) {
            int n = ps.length;
            if (n == 0) return;
            double[] px = new double[n], py = new double[n], pm = new double[n];
            for (int i = 0; i < n; i++) {
                px[i] = ps[i].x; py[i] = ps[i].y; pm[i] = ps[i].mass;
            }
            double eps2 = cfg.softening * cfg.softening;
            double gConst = cfg.gravitationalConstant;

            if (n < PARALLEL_BODY_THRESHOLD) {
                for (int i = 0; i < n; i++) accelerate(ps, i, px, py, pm, eps2, gConst);
            } else {
                IntStream.range(0, n).parallel()
                        .forEach(i -> accelerate(ps, i, px, py, pm, eps2, gConst));
            }
        }

        /** Total gravitational acceleration on body i. Writes only to ps[i]. */
        private static void accelerate(Planet[] ps, int i, double[] px, double[] py, double[] pm,
                                       double eps2, double gConst) {
            Planet a = ps[i];
            if (a.fixed) { a.ax = 0; a.ay = 0; return; }  // pinned: acceleration unused
            double xi = px[i], yi = py[i], ax = 0, ay = 0;
            for (int j = 0; j < px.length; j++) {
                if (j == i) continue;
                double dx = px[j] - xi, dy = py[j] - yi;
                double d2 = dx * dx + dy * dy + eps2;
                // a = G m_j r / |r|^3 — the pulled body's own mass cancels out.
                double s = gConst * pm[j] / (d2 * Math.sqrt(d2));
                ax += s * dx; ay += s * dy;
            }
            a.ax = ax; a.ay = ay;
        }

        /** Impulse-based elastic/inelastic collision between two circles. */
        private void resolveCollision(Planet a, Planet b, double restitution) {
            double dx = b.x - a.x, dy = b.y - a.y;
            double dist = Math.sqrt(dx * dx + dy * dy);
            double minDist = a.radius + b.radius;
            if (dist >= minDist || dist == 0) return;

            double nx = dx / dist, ny = dy / dist;

            // Separate overlapping bodies proportionally to inverse mass. Two fixed
            // bodies have nothing to give, so leave them overlapping.
            double overlap = minDist - dist;
            double invA = a.invMass(), invB = b.invMass();
            double invSum = invA + invB;
            if (invSum == 0) return;
            a.x -= nx * overlap * (invA / invSum);
            a.y -= ny * overlap * (invA / invSum);
            b.x += nx * overlap * (invB / invSum);
            b.y += ny * overlap * (invB / invSum);

            // Relative velocity along the collision normal.
            double rvx = b.vx - a.vx, rvy = b.vy - a.vy;
            double velAlongNormal = rvx * nx + rvy * ny;
            if (velAlongNormal > 0) return; // already separating

            // Impulse magnitude (conserves momentum; restitution controls energy).
            double jImpulse = -(1 + restitution) * velAlongNormal / invSum;
            double jx = jImpulse * nx, jy = jImpulse * ny;
            a.vx -= jx * invA;  a.vy -= jy * invA;
            b.vx += jx * invB;  b.vy += jy * invB;
        }

        // ----------------- Rendering -----------------
        @Override protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            if (cfg.showGravityWell) {
                paintGravityWell(g2);
            } else {
                // Work in plane coordinates: the square box is centred in the panel.
                Graphics2D gt = (Graphics2D) g2.create();
                gt.translate(boxOriginX(), boxOriginY());
                paintFlat(gt);
                gt.dispose();
            }

            g2.setColor(new Color(200, 210, 255, 180));
            g2.drawString(String.format("Bodies: %d   %s   (drag to launch a body)%s",
                    planets.size(), cfg.paused ? "PAUSED" : "running",
                    cfg.showGravityWell ? "   [gravitational well]" : ""), 12, 20);
        }

        /** The top-down 2D view, drawn in plane coordinates. */
        private void paintFlat(Graphics2D g2) {
            int box = boxSize();
            Planet[] snap = snapshotForRender();
            prepareTrails(snap, null);

            // Trails
            for (Planet p : snap) drawTrail(g2, p);

            // Planets
            for (Planet p : snap) {
                double radius = p.radius;
                int d = (int) (radius * 2);
                // soft glow
                g2.setColor(new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(), 40));
                g2.fillOval((int) (p.x - radius * 1.8), (int) (p.y - radius * 1.8),
                        (int) (d * 1.8), (int) (d * 1.8));
                g2.setColor(p.color);
                g2.fillOval((int) (p.x - radius), (int) (p.y - radius), d, d);
                g2.setColor(p.color.brighter());
                g2.drawOval((int) (p.x - radius), (int) (p.y - radius), d, d);
                if (p.fixed) drawAnchor(g2, p.x, p.y, radius);

                // velocity vector
                if (cfg.showVectors) {
                    g2.setColor(Color.WHITE);
                    g2.drawLine((int) p.x, (int) p.y,
                            (int) (p.x + p.vx * 0.3), (int) (p.y + p.vy * 0.3));
                }
            }

            // Launch preview: ghost body, a faint guide out to the cursor, and the
            // arrow showing the velocity it will actually be launched with.
            if (dragging) {
                double[] v = dragVelocity();
                double r = cfg.spawnRadius;
                g2.setColor(new Color(255, 255, 255, 120));
                g2.drawLine((int) dragFromX, (int) dragFromY, (int) dragToX, (int) dragToY);
                g2.setColor(new Color(255, 255, 255, 70));
                g2.fillOval((int) (dragFromX - r), (int) (dragFromY - r), (int) (r * 2), (int) (r * 2));
                g2.setColor(new Color(255, 255, 255, 200));
                g2.drawOval((int) (dragFromX - r), (int) (dragFromY - r), (int) (r * 2), (int) (r * 2));
                if (cfg.spawnFixed) drawAnchor(g2, dragFromX, dragFromY, r);
                else drawArrow(g2, dragFromX, dragFromY, dragFromX + v[0] * 0.3, dragFromY + v[1] * 0.3);
            }

            // Walls of the box
            g2.setColor(new Color(90, 110, 200));
            g2.setStroke(new BasicStroke(3));
            g2.drawRect(1, 1, box - 3, box - 3);
            g2.setStroke(new BasicStroke(1));
        }

        // ----------------- Trails -----------------
        //
        // At a few hundred bodies the trails are the most expensive thing in a frame:
        // tens of thousands of short segments, each needing two sheet lookups and a
        // projection before it can be drawn. That maths is done up front in parallel
        // into reusable coordinate buffers, leaving the draw pass to do nothing but
        // issue lines. (Batching the segments into Path2D objects was tried and lost
        // to plain drawLine — the path machinery costs more than it saves here.)

        /**
         * Bodies in a stable order, each stamped with the slot holding its prepared
         * render data. Taken once per paint so the two passes agree even if the user
         * launches a body mid-frame.
         */
        private Planet[] snapshotForRender() {
            Planet[] snap = planets.toArray(new Planet[0]);
            for (int i = 0; i < snap.length; i++) snap[i].renderIndex = i;
            return snap;
        }

        private static final int TRAIL_FADE_STEPS = 24;
        private static final int PARALLEL_TRAIL_THRESHOLD = 3_000;   // total trail points

        private int[][] trailX = new int[0][], trailY = new int[0][];
        private int[] trailCount = new int[0];
        private Color[][] trailRamp = new Color[0][];

        /**
         * Projects every trail into its own coordinate buffer, in parallel across
         * bodies — each writes only its own slot. Passing a WellView projects onto the
         * 3D sheet; passing null keeps the flat top-down view.
         */
        private void prepareTrails(Planet[] snap, WellView v) {
            int n = snap.length;
            if (trailX.length < n) {
                trailX = Arrays.copyOf(trailX, n);
                trailY = Arrays.copyOf(trailY, n);
                trailRamp = Arrays.copyOf(trailRamp, n);
                trailCount = Arrays.copyOf(trailCount, n);
            }
            if (!cfg.showTrails) {
                Arrays.fill(trailCount, 0, n, 0);
                return;
            }
            long points = 0;
            for (Planet p : snap) points += p.trail.size();
            if (points < PARALLEL_TRAIL_THRESHOLD) {
                for (int i = 0; i < n; i++) buildTrail(i, snap[i], v);
            } else {
                IntStream.range(0, n).parallel().forEach(i -> buildTrail(i, snap[i], v));
            }
        }

        /** Fills slot i with p's projected trail, growing its buffers if needed. */
        private void buildTrail(int i, Planet p, WellView v) {
            int m = p.trail.size();
            trailCount[i] = m;
            if (m < 2) return;
            if (trailX[i] == null || trailX[i].length < m) {
                trailX[i] = new int[m + 64];
                trailY[i] = new int[m + 64];
            }
            int[] xs = trailX[i], ys = trailY[i];
            int k = 0;
            for (Planet.Point2D pt : p.trail) {
                if (v == null) {
                    xs[k] = (int) pt.x(); ys[k] = (int) pt.y();
                } else {
                    xs[k] = (int) v.sx(pt.x(), pt.y());
                    ys[k] = (int) v.sy(pt.x(), pt.y(), sampleWell(pt.x(), pt.y()));
                }
                k++;
            }
            // The fade is quantised, so a body needs a couple of dozen Colors rather
            // than one per segment — and the ramp survives as long as its body does.
            Color[] ramp = trailRamp[i];
            if (ramp == null || ramp[0] == null || ramp[TRAIL_FADE_STEPS - 1].getRGB() != rampTop(p)) {
                ramp = trailRamp[i] = new Color[TRAIL_FADE_STEPS];
                for (int s = 0; s < TRAIL_FADE_STEPS; s++) {
                    int alpha = (int) ((s + 0.5f) / TRAIL_FADE_STEPS * 0.6f * 255);
                    ramp[s] = new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(), alpha);
                }
            }
        }

        /** The RGBA the brightest step of p's ramp should have — used to spot a stale ramp. */
        private static int rampTop(Planet p) {
            int alpha = (int) ((TRAIL_FADE_STEPS - 0.5f) / TRAIL_FADE_STEPS * 0.6f * 255);
            return (alpha << 24) | (p.color.getRGB() & 0xFFFFFF);
        }

        /** Draws a prepared trail, fading along its length. */
        private void drawTrail(Graphics2D g2, Planet p) {
            int i = p.renderIndex, m = trailCount[i];
            if (m < 2) return;
            int[] xs = trailX[i], ys = trailY[i];
            Color[] ramp = trailRamp[i];
            int step = -1;
            for (int k = 1; k < m; k++) {
                int s = Math.min(TRAIL_FADE_STEPS - 1, k * TRAIL_FADE_STEPS / m);
                if (s != step) { step = s; g2.setColor(ramp[s]); }
                g2.drawLine(xs[k - 1], ys[k - 1], xs[k], ys[k]);
            }
        }

        /** Marks a pinned body: a ring with four ticks, like a surveyor's mark. */
        private void drawAnchor(Graphics2D g2, double cx, double cy, double r) {
            double ring = r + 5;
            g2.setColor(new Color(235, 240, 255, 190));
            g2.drawOval((int) (cx - ring), (int) (cy - ring), (int) (ring * 2), (int) (ring * 2));
            for (int k = 0; k < 4; k++) {
                double ang = Math.PI / 4 + k * Math.PI / 2;
                double ux = Math.cos(ang), uy = Math.sin(ang);
                g2.drawLine((int) (cx + ux * ring), (int) (cy + uy * ring),
                        (int) (cx + ux * (ring + 4)), (int) (cy + uy * (ring + 4)));
            }
        }

        /** A line with a small arrowhead, used for the launch preview. */
        private void drawArrow(Graphics2D g2, double x0, double y0, double x1, double y1) {
            g2.setColor(new Color(255, 255, 255, 200));
            g2.drawLine((int) x0, (int) y0, (int) x1, (int) y1);
            double dx = x1 - x0, dy = y1 - y0;
            double len = Math.hypot(dx, dy);
            if (len < 6) return;
            double ux = dx / len, uy = dy / len, head = Math.min(12, len * 0.35);
            g2.drawLine((int) x1, (int) y1,
                    (int) (x1 - head * (ux * 0.87 - uy * 0.5)), (int) (y1 - head * (uy * 0.87 + ux * 0.5)));
            g2.drawLine((int) x1, (int) y1,
                    (int) (x1 - head * (ux * 0.87 + uy * 0.5)), (int) (y1 - head * (uy * 0.87 - ux * 0.5)));
        }

        // ----------------- Gravitational well (3D) -----------------
        //
        // The box is treated as a rubber sheet: every point of the (x,y) plane is
        // displaced along z by the local Newtonian potential of all the planets,
        //     Phi(x,y) = -sum_i G m_i / |r - r_i|,
        // so heavy bodies punch deep funnels into the sheet, and nearby bodies'
        // wells merge into a shared basin. The sheet is sampled on a regular grid,
        // then drawn as shaded quads through an oblique camera (tilt + rotation),
        // with each planet drawn as a lit sphere sitting in the sheet above its dip.

        private double[][] wellZ;   // sheet height, [n+1][n+1] — the plane is square
        private int wellN;          // mesh cells per side
        private double wellCell;    // world units per cell

        /** Maximum dip, in world units, that the sheet is allowed to reach. */
        private double maxWellDepth() { return boxSize() * 0.45; }

        /**
         * Squared softening length used when sampling a body's potential. Widened
         * beyond the body's own radius, and floored at roughly one mesh cell, so each
         * funnel spans several cells and reads as a bowl rather than a one-cell spike.
         */
        private double wellSoftening(Planet p) {
            double soft = Math.max(Math.max(p.radius * 2.0, cfg.softening), wellCell * 0.8);
            return soft * soft;
        }

        /** Samples the gravitational potential onto the mesh grid. */
        private void computeWellGrid(int box) {
            int n = Math.max(8, cfg.wellResolution);
            if (wellZ == null || wellN != n) {
                wellN = n;
                wellZ = new double[n + 1][n + 1];
            }
            wellCell = (double) box / n;

            double maxZ = maxWellDepth();
            double k = cfg.wellDepthScale;
            Planet[] ps = planets.toArray(new Planet[0]);
            int bodies = ps.length;
            double[] bx = new double[bodies], by = new double[bodies];
            double[] bm = new double[bodies], beps = new double[bodies];
            for (int m = 0; m < bodies; m++) {
                bx[m] = ps[m].x; by[m] = ps[m].y;
                bm[m] = ps[m].mass; beps[m] = wellSoftening(ps[m]);
            }

            // Each column owns its own row of the array, so the sampling sweep — the
            // heaviest part of a frame once there are many bodies — fans out across
            // cores with no synchronisation at all.
            double cell = wellCell;
            if ((long) (n + 1) * (n + 1) * Math.max(1, bodies) < PARALLEL_SAMPLE_THRESHOLD) {
                for (int i = 0; i <= n; i++) sampleColumn(i, n, cell, bx, by, bm, beps, maxZ, k);
            } else {
                IntStream.rangeClosed(0, n).parallel()
                        .forEach(i -> sampleColumn(i, n, cell, bx, by, bm, beps, maxZ, k));
            }
        }

        /** Fills one column of the sheet. Writes only to wellZ[i]. */
        private void sampleColumn(int i, int n, double cell, double[] bx, double[] by,
                                  double[] bm, double[] beps, double maxZ, double k) {
            double gx = i * cell;
            double[] column = wellZ[i];
            for (int j = 0; j <= n; j++) {
                double gy = j * cell;
                double potential = 0;
                for (int m = 0; m < bx.length; m++) {
                    double dx = gx - bx[m], dy = gy - by[m];
                    potential += bm[m] / Math.sqrt(dx * dx + dy * dy + beps[m]);
                }
                column[j] = depthOf(potential, maxZ, k);
            }
        }

        /**
         * Saturating (exponential) mapping of potential to sheet height: linear for
         * shallow wells, asymptotic towards maxZ so deep wells taper off smoothly
         * instead of clipping to a flat floor.
         */
        private static double depthOf(double potential, double maxZ, double k) {
            return -maxZ * (1 - Math.exp(-potential * k / maxZ));
        }

        /**
         * Height of the sheet at a planet, ignoring the planet's own contribution —
         * i.e. the background curvature it is resting in. Using this (rather than the
         * bottom of its own funnel) keeps the body visible above the dip it creates,
         * which is also how the rubber-sheet analogy is normally drawn.
         */
        private double restingZ(Planet self) {
            double potential = 0;
            for (Planet p : planets) {
                if (p == self) continue;
                double dx = self.x - p.x, dy = self.y - p.y;
                potential += p.mass / Math.sqrt(dx * dx + dy * dy + wellSoftening(p));
            }
            return depthOf(potential, maxWellDepth(), cfg.wellDepthScale);
        }

        /** Bilinear lookup of the sheet height at an arbitrary point. */
        private double sampleWell(double x, double y) {
            if (wellZ == null) return 0;
            double fx = Math.max(0, Math.min(wellN, x / wellCell));
            double fy = Math.max(0, Math.min(wellN, y / wellCell));
            int i = Math.min((int) fx, wellN - 1);
            int j = Math.min((int) fy, wellN - 1);
            double tx = fx - i, ty = fy - j;
            double top = wellZ[i][j] * (1 - tx) + wellZ[i + 1][j] * tx;
            double bot = wellZ[i][j + 1] * (1 - tx) + wellZ[i + 1][j + 1] * tx;
            return top * (1 - ty) + bot * ty;
        }

        // Mesh geometry and shading, rebuilt each frame ahead of the draw pass.
        // gridSx[j][i] / gridSy[j][i] are the screen coordinates of grid point (i,j);
        // cellFill / cellLine are indexed j * wellN + i.
        private int[][] gridSx, gridSy;
        private Color[] cellFill, cellLine;

        /**
         * Projects every grid point and shades every cell, in parallel across rows.
         * Rows are independent — each writes only its own slice — and doing this up
         * front also removes the fourfold redundancy of projecting shared corners
         * once per adjoining cell.
         */
        private void prepareMesh(WellView v, double maxZ) {
            int n = wellN;
            if (gridSx == null || gridSx.length != n + 1) {
                gridSx = new int[n + 1][n + 1];
                gridSy = new int[n + 1][n + 1];
                cellFill = new Color[n * n];
                cellLine = new Color[n * n];
            }
            if (n * n < PARALLEL_CELL_THRESHOLD) {
                for (int j = 0; j <= n; j++) prepareMeshRow(j, n, v, maxZ);
            } else {
                IntStream.rangeClosed(0, n).parallel().forEach(j -> prepareMeshRow(j, n, v, maxZ));
            }
        }

        /** Projects row j of grid points, and shades the cells hanging below it. */
        private void prepareMeshRow(int j, int n, WellView v, double maxZ) {
            double cell = wellCell;
            double y = j * cell;
            int[] sxRow = gridSx[j], syRow = gridSy[j];
            for (int i = 0; i <= n; i++) {
                double x = i * cell;
                sxRow[i] = (int) v.sx(x, y);
                syRow[i] = (int) v.sy(x, y, wellZ[i][j]);
            }
            if (j == n) return;   // last grid row bounds no cells of its own

            int base = j * n;
            for (int i = 0; i < n; i++) {
                double z00 = wellZ[i][j], z10 = wellZ[i + 1][j];
                double z01 = wellZ[i][j + 1], z11 = wellZ[i + 1][j + 1];
                double avgZ = (z00 + z10 + z01 + z11) * 0.25;
                double depth = Math.min(1, -avgZ / maxZ);
                // Diffuse shading from the sheet's own slope.
                double dzdx = ((z10 + z11) - (z00 + z01)) * 0.5 / cell;
                double dzdy = ((z01 + z11) - (z00 + z10)) * 0.5 / cell;
                double len = Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1);
                double lambert = (0.45 * dzdx + 0.62 * dzdy + 0.65) / len;
                double shade = 0.35 + 0.75 * Math.max(0, lambert);
                cellFill[base + i] = wellColor(depth, shade, false);
                cellLine[base + i] = wellColor(depth, shade, true);
            }
        }

        private void paintGravityWell(Graphics2D g2) {
            int box = boxSize();
            if (box <= 0) return;

            computeWellGrid(box);
            WellView v = new WellView();
            double maxZ = maxWellDepth();
            Planet[] snap = snapshotForRender();
            prepareTrails(snap, v);

            // Bucket the planets by mesh row so they can be drawn interleaved with
            // the sheet — that gives correct occlusion without a depth buffer.
            List<List<Planet>> byRow = new ArrayList<>(wellN);
            for (int j = 0; j < wellN; j++) byRow.add(null);
            for (Planet p : snap) {
                int j = Math.max(0, Math.min(wellN - 1, (int) (p.y / wellCell)));
                if (byRow.get(j) == null) byRow.set(j, new ArrayList<>(2));
                byRow.get(j).add(p);
            }

            // Everything that can be worked out ahead of the draw calls is worked out
            // in parallel; the serial pass below is then nothing but Java2D calls.
            prepareMesh(v, maxZ);

            // Painter's algorithm: rows always run far -> near (cos(yaw) > 0);
            // columns run far -> near depending on the sign of sin(yaw).
            boolean colsAscending = v.sinY >= 0;
            int[] xs = new int[4], ys = new int[4];

            // The mesh quads tile the plane edge to edge, so antialiasing their shared
            // borders costs a lot of rasteriser time and only muddies the seams. The
            // spheres, trails and rim below keep it.
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

            for (int j = 0; j < wellN; j++) {
                int[] rowSx = gridSx[j], rowSy = gridSy[j];
                int[] nextSx = gridSx[j + 1], nextSy = gridSy[j + 1];
                int base = j * wellN;
                for (int c = 0; c < wellN; c++) {
                    int i = colsAscending ? c : wellN - 1 - c;
                    xs[0] = rowSx[i];      ys[0] = rowSy[i];
                    xs[1] = rowSx[i + 1];  ys[1] = rowSy[i + 1];
                    xs[2] = nextSx[i + 1]; ys[2] = nextSy[i + 1];
                    xs[3] = nextSx[i];     ys[3] = nextSy[i];

                    g2.setColor(cellFill[base + i]);
                    g2.fillPolygon(xs, ys, 4);
                    // Only the top and left edges: the neighbouring cells supply the
                    // rest, and the outer two edges are covered by the rim pass.
                    g2.setColor(cellLine[base + i]);
                    g2.drawLine(xs[0], ys[0], xs[1], ys[1]);
                    g2.drawLine(xs[0], ys[0], xs[3], ys[3]);
                }

                List<Planet> here = byRow.get(j);
                if (here != null) {
                    if (here.size() > 1) {
                        here.sort(colsAscending ? Comparator.comparingDouble(p -> p.x)
                                                : Comparator.comparingDouble(p -> -p.x));
                    }
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);
                    for (Planet p : here) drawPlanet3D(g2, v, p);
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_OFF);
                }
            }
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Rim of the box, riding the edge of the sheet.
            g2.setColor(new Color(120, 145, 235, 200));
            g2.setStroke(new BasicStroke(2f));
            drawWellEdge(g2, 0, 0, 1, 0);
            drawWellEdge(g2, 0, wellN, 1, 0);
            drawWellEdge(g2, 0, 0, 0, 1);
            drawWellEdge(g2, wellN, 0, 0, 1);
            g2.setStroke(new BasicStroke(1f));

            // Launch preview: a ghost body sitting on the sheet, plus its heading.
            if (dragging) {
                double r = cfg.spawnRadius;
                double z = sampleWell(dragFromX, dragFromY) + r;
                double cx = v.sx(dragFromX, dragFromY), cy = v.sy(dragFromX, dragFromY, z);
                double rr = r * v.scale;
                g2.setColor(new Color(255, 255, 255, 120));
                g2.drawLine((int) cx, (int) cy, (int) v.sx(dragToX, dragToY),
                        (int) v.sy(dragToX, dragToY, sampleWell(dragToX, dragToY)));
                g2.setColor(new Color(255, 255, 255, 70));
                g2.fillOval((int) (cx - rr), (int) (cy - rr), (int) (rr * 2), (int) (rr * 2));
                g2.setColor(new Color(255, 255, 255, 200));
                g2.drawOval((int) (cx - rr), (int) (cy - rr), (int) (rr * 2), (int) (rr * 2));
                if (cfg.spawnFixed) {
                    drawAnchor(g2, cx, cy, rr);
                } else {
                    double[] vel = dragVelocity();
                    double tx = dragFromX + vel[0] * 0.3, ty = dragFromY + vel[1] * 0.3;
                    drawArrow(g2, cx, cy, v.sx(tx, ty), v.sy(tx, ty, z));
                }
            }
        }

        /** Traces one edge of the mesh, following the sheet's height. */
        private void drawWellEdge(Graphics2D g2, int i0, int j0, int di, int dj) {
            int px = 0, py = 0;
            for (int s = 0; s <= wellN; s++) {
                int i = i0 + di * s, j = j0 + dj * s;
                int sx = gridSx[j][i], sy = gridSy[j][i];
                if (s > 0) g2.drawLine(px, py, sx, sy);
                px = sx; py = sy;
            }
        }

        /** Sheet colouring: dark blue where flat, hot violet where deeply curved. */
        private Color wellColor(double depth, double shade, boolean line) {
            double t = Math.max(0, Math.min(1, depth));
            double r, g, b;
            if (line) {
                r = 60 + 195 * t; g = 85 + 65 * t; b = 155 + 100 * t;
                shade = 0.6 + 0.5 * shade;
            } else {
                r = 16 + 170 * t; g = 22 + 48 * t; b = 52 + 155 * t;
            }
            return new Color(channel(r * shade), channel(g * shade), channel(b * shade));
        }

        private static int channel(double v) { return (int) Math.max(0, Math.min(255, v)); }

        /** A planet as a lit sphere sitting at the bottom of its own well. */
        private void drawPlanet3D(Graphics2D g2, WellView v, Planet p) {
            double radius = p.radius;
            double z = restingZ(p) + radius;
            double cx = v.sx(p.x, p.y);
            double cy = v.sy(p.x, p.y, z);
            double r = radius * v.scale;

            // Trail, projected down onto the sheet.
            drawTrail(g2, p);

            // Sphere: radial gradient with the highlight up and to the left.
            g2.setPaint(new RadialGradientPaint(
                    new java.awt.geom.Point2D.Double(cx - r * 0.35, cy - r * 0.4),
                    (float) Math.max(1, r * 1.35),
                    new float[]{0f, 0.55f, 1f},
                    new Color[]{p.color.brighter().brighter(), p.color, p.color.darker().darker()}));
            g2.fillOval((int) (cx - r), (int) (cy - r), (int) (r * 2), (int) (r * 2));
            g2.setPaint(null);
            g2.setColor(new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(), 140));
            g2.drawOval((int) (cx - r), (int) (cy - r), (int) (r * 2), (int) (r * 2));
            if (p.fixed) drawAnchor(g2, cx, cy, r);

            // Stalk dropping from the sphere to the floor of the well it digs.
            g2.setColor(new Color(255, 255, 255, 40));
            g2.drawLine((int) cx, (int) (cy + r),
                    (int) cx, (int) v.sy(p.x, p.y, sampleWell(p.x, p.y)));

            if (cfg.showVectors) {
                double tx = p.x + p.vx * 0.3, ty = p.y + p.vy * 0.3;
                g2.setColor(Color.WHITE);
                g2.drawLine((int) cx, (int) cy, (int) v.sx(tx, ty), (int) v.sy(tx, ty, z));
            }
        }

        /**
         * Oblique camera over the (x,y) plane with a vertical z axis: the plane is
         * rotated by `yaw` about the vertical, then tilted by `pitch` towards the
         * viewer. pitch = 0 is straight down (no z visible), pitch -> 90 is edge-on.
         */
        final class WellView {
            final double box, half;
            final double cosP, sinP, cosY, sinY, scale, cx, cy;

            WellView() {
                box = boxSize();
                half = box / 2.0;
                double pitch = Math.toRadians(cfg.wellPitch);
                double yaw = Math.toRadians(cfg.wellYaw);
                cosP = Math.cos(pitch); sinP = Math.sin(pitch);
                cosY = Math.cos(yaw);   sinY = Math.sin(yaw);

                // A square rotated by `yaw` spans this much in each direction, and the
                // drawing reaches a well's depth below the plane. Scale to whichever of
                // the two axes is tighter, then centre the whole figure in the panel.
                // Wells approach maxWellDepth only asymptotically, so reserving the
                // full depth would leave the sheet stranded at the top of the panel;
                // budget for a deep-but-realistic well instead.
                int pw = getWidth(), ph = getHeight();
                double footprint = box * (Math.abs(cosY) + Math.abs(sinY));
                double tall = footprint * cosP + maxWellDepth() * 0.6 * sinP;
                scale = Math.min(0.94 * pw / footprint, 0.94 * ph / tall);
                cx = pw / 2.0;
                cy = (ph - tall * scale) / 2.0 + footprint * cosP * scale / 2.0;
            }

            /** Screen x of a plane point. */
            double sx(double x, double y) {
                return cx + ((x - half) * cosY - (y - half) * sinY) * scale;
            }

            /** Screen y of a plane point lifted to height z. */
            double sy(double x, double y, double z) {
                double depth = (x - half) * sinY + (y - half) * cosY;
                return cy + (depth * cosP - z * sinP) * scale;
            }

            /** Screen point back onto the z = 0 plane (used for drag-to-launch). */
            double[] unproject(double screenX, double screenY) {
                double xr = (screenX - cx) / scale;
                double yr = (screenY - cy) / (scale * Math.max(1e-6, cosP));
                return new double[]{
                        xr * cosY + yr * sinY + half,
                        -xr * sinY + yr * cosY + half};
            }
        }
    }

    // ------------------------------------------------------------------
    // The runtime control panel.
    // ------------------------------------------------------------------
    static class ControlPanel extends JPanel {
        private final SimulationConfig cfg;
        private final SimulationPanel sim;

        ControlPanel(SimulationConfig cfg, SimulationPanel sim) {
            this.cfg = cfg;
            this.sim = sim;
            setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
            setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

            // --- Physics group ---
            JPanel physics = group("Physics");
            physics.add(slider("Gravity (G)", 0, 10000, (int) cfg.gravitationalConstant,
                    v -> cfg.gravitationalConstant = v, "%.0f", 1));
            physics.add(slider("Time step (dt ×1000)", 1, 30, (int) (cfg.timeStep * 1000),
                    v -> cfg.timeStep = v / 1000.0, "%.3f", 0.001));
            physics.add(slider("Speed (steps/frame)", 1, 20, cfg.stepsPerFrame,
                    v -> cfg.stepsPerFrame = (int) v, "%.0f", 1));
            physics.add(slider("Softening (px)", 0, 30, (int) cfg.softening,
                    v -> cfg.softening = v, "%.0f", 1));
            add(physics);

            // --- Collisions group ---
            JPanel collisions = group("Collisions & Walls");
            collisions.add(slider("Wall bounciness (% speed kept)", 0, 100, (int) (cfg.wallRestitution * 100),
                    v -> cfg.wallRestitution = v / 100.0, "%.2f", 0.01));

            JCheckBox collideBox = new JCheckBox("Planet-planet collisions", cfg.planetCollisions);
            collideBox.addActionListener(e -> cfg.planetCollisions = collideBox.isSelected());
            collideBox.setAlignmentX(LEFT_ALIGNMENT);
            collideBox.setEnabled(!cfg.pointMasses);

            JCheckBox pointBox = new JCheckBox("Zero radius (no collisions)", cfg.pointMasses);
            pointBox.setToolTipText("Treat every body as a dimensionless point for contact: they pass "
                    + "straight through each other. Still drawn full size, and still bounce off the walls.");
            pointBox.setAlignmentX(LEFT_ALIGNMENT);
            pointBox.addActionListener(e -> {
                cfg.pointMasses = pointBox.isSelected();
                collideBox.setEnabled(!cfg.pointMasses);
                sim.repaint();
            });
            collisions.add(pointBox);

            collisions.add(collideBox);
            collisions.add(slider("Planet bounciness (%)", 0, 100, (int) (cfg.planetRestitution * 100),
                    v -> cfg.planetRestitution = v / 100.0, "%.2f", 0.01));
            add(collisions);

            // --- Spawn settings group ---
            JPanel spawn = group("Next Body (drag on the canvas)");
            spawn.add(slider("Mass", 10, 5000, (int) cfg.spawnMass,
                    v -> cfg.spawnMass = v, "%.0f", 1));
            spawn.add(slider("Radius", 2, 50, (int) cfg.spawnRadius,
                    v -> cfg.spawnRadius = v, "%.0f", 1));
            spawn.add(slider("Launch speed", 0, 300, (int) cfg.spawnSpeed,
                    v -> cfg.spawnSpeed = v, "%.0f", 1));
            JCheckBox fixedBox = new JCheckBox("Fix position", cfg.spawnFixed);
            fixedBox.setToolTipText("Pin the body where it is placed: it pulls on everything "
                    + "else but nothing can move it. Its launch direction is ignored.");
            fixedBox.setAlignmentX(LEFT_ALIGNMENT);
            fixedBox.addActionListener(e -> {
                cfg.spawnFixed = fixedBox.isSelected();
                sim.repaint();
            });
            spawn.add(fixedBox);
            add(spawn);

            // --- Display group ---
            JPanel display = group("Display");
            JCheckBox trails = new JCheckBox("Show trails", cfg.showTrails);
            trails.addActionListener(e -> {
                cfg.showTrails = trails.isSelected();
                if (!cfg.showTrails) sim.clearTrails();
            });
            trails.setAlignmentX(LEFT_ALIGNMENT);
            display.add(trails);
            display.add(slider("Trail length", 10, 1000, cfg.trailLength,
                    v -> cfg.trailLength = (int) v, "%.0f", 1));
            JCheckBox vectors = new JCheckBox("Show velocity vectors", cfg.showVectors);
            vectors.addActionListener(e -> cfg.showVectors = vectors.isSelected());
            vectors.setAlignmentX(LEFT_ALIGNMENT);
            display.add(vectors);
            add(display);

            // --- Gravitational well group ---
            JPanel well = group("Gravitational Well");
            JCheckBox wellBox = new JCheckBox("Show gravitational well", cfg.showGravityWell);
            wellBox.setToolTipText("Render the box in 3D: planets as spheres on the (x,y) plane, "
                    + "z showing the space-time well their mass creates");
            wellBox.setAlignmentX(LEFT_ALIGNMENT);
            wellBox.addActionListener(e -> {
                cfg.showGravityWell = wellBox.isSelected();
                sim.repaint();
            });
            well.add(wellBox);
            well.add(slider("Well depth (×100)", 0, 800, (int) (cfg.wellDepthScale * 100),
                    v -> cfg.wellDepthScale = v / 100.0, "%.2f", 0.01));
            well.add(slider("Camera tilt (°)", 10, 85, (int) cfg.wellPitch,
                    v -> cfg.wellPitch = v, "%.0f", 1));
            well.add(slider("Camera rotation (°)", -60, 60, (int) cfg.wellYaw,
                    v -> cfg.wellYaw = v, "%.0f", 1));
            well.add(slider("Mesh resolution", 12, 96, cfg.wellResolution,
                    v -> cfg.wellResolution = (int) v, "%.0f", 1));
            add(well);

            // --- Buttons ---
            JPanel buttons = new JPanel(new GridLayout(0, 1, 4, 4));
            buttons.setAlignmentX(LEFT_ALIGNMENT);
            buttons.setBorder(BorderFactory.createEmptyBorder(8, 0, 0, 0));

            JButton pauseBtn = new JButton("Pause");
            pauseBtn.addActionListener(e -> {
                cfg.paused = !cfg.paused;
                pauseBtn.setText(cfg.paused ? "Resume" : "Pause");
            });
            buttons.add(pauseBtn);

            JButton clearBtn = new JButton("Clear all bodies");
            clearBtn.addActionListener(e -> sim.clearPlanets());
            buttons.add(clearBtn);

            JButton clearTrailsBtn = new JButton("Clear trails");
            clearTrailsBtn.addActionListener(e -> sim.clearTrails());
            buttons.add(clearTrailsBtn);

            add(buttons);
            add(Box.createVerticalGlue());
        }

        private JPanel group(String title) {
            JPanel p = new JPanel();
            p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
            p.setBorder(BorderFactory.createTitledBorder(
                    BorderFactory.createEtchedBorder(), title,
                    TitledBorder.LEFT, TitledBorder.TOP));
            p.setAlignmentX(LEFT_ALIGNMENT);
            return p;
        }

        /**
         * Builds a labelled slider whose current value is displayed live.
         * The consumer receives the raw slider value scaled by `scale`.
         */
        private JPanel slider(String name, int min, int max, int initial,
                              java.util.function.DoubleConsumer onChange,
                              String fmt, double scale) {
            JPanel row = new JPanel(new BorderLayout(4, 0));
            row.setAlignmentX(LEFT_ALIGNMENT);
            row.setMaximumSize(new Dimension(280, 48));

            JLabel label = new JLabel(name + ": " + String.format(fmt, initial * scale));
            JSlider s = new JSlider(min, max, Math.max(min, Math.min(max, initial)));
            s.addChangeListener(e -> {
                double v = s.getValue();
                onChange.accept(v);
                label.setText(name + ": " + String.format(fmt, v * scale));
            });
            row.add(label, BorderLayout.NORTH);
            row.add(s, BorderLayout.CENTER);
            return row;
        }
    }
}