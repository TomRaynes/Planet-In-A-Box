package org.example;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.util.List;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * PlanetBoxSimulation
 *
 * A 2D Newtonian gravity simulation of planets confined to an inescapable box.
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
 *   - Number of planets, mass range and radius range used on reset
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
        volatile boolean showGravityWell = false;
        volatile double wellDepthScale = 3.5;   // px of dip per unit of potential
        volatile double wellPitch = 58;         // camera tilt, degrees (0 = top-down)
        volatile double wellYaw = 22;           // camera rotation about the vertical, degrees
        volatile int wellResolution = 64;       // mesh columns across the box

        // Used when (re)spawning planets:
        volatile int planetCount = 8;
        volatile double minMass = 50, maxMass = 800;
        volatile double minRadius = 6, maxRadius = 18;
        volatile double maxInitialSpeed = 60;
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
        final Deque<Point2D> trail = new ArrayDeque<>();

        Planet(double x, double y, double vx, double vy, double mass, double radius, Color color) {
            this.x = x; this.y = y;
            this.vx = vx; this.vy = vy;
            this.mass = mass; this.radius = radius;
            this.color = color;
        }

        record Point2D(double x, double y) {}
    }

    // ------------------------------------------------------------------
    // The simulation canvas: physics loop + rendering.
    // ------------------------------------------------------------------
    static class SimulationPanel extends JPanel {
        private final SimulationConfig cfg;
        private final List<Planet> planets = new CopyOnWriteArrayList<>();
        private final Random rng = new Random();
        private javax.swing.Timer timer;

        SimulationPanel(SimulationConfig cfg) {
            this.cfg = cfg;
            setPreferredSize(new Dimension(900, 700));
            setBackground(new Color(8, 10, 24));
            respawnPlanets();

            // Click to add a planet at the cursor with a random velocity.
            addMouseListener(new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) {
                    double x = e.getX(), y = e.getY();
                    if (cfg.showGravityWell) {
                        // Map the click back onto the (x,y) plane through the 3D camera.
                        double[] world = new WellView(getWidth(), getHeight()).unproject(x, y);
                        x = world[0]; y = world[1];
                    }
                    x = Math.max(0, Math.min(getWidth(), x));
                    y = Math.max(0, Math.min(getHeight(), y));
                    planets.add(randomPlanet(x, y));
                }
            });
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

        void respawnPlanets() {
            planets.clear();
            int w = Math.max(getWidth(), 900), h = Math.max(getHeight(), 700);
            for (int i = 0; i < cfg.planetCount; i++) {
                double r = cfg.minRadius + rng.nextDouble() * (cfg.maxRadius - cfg.minRadius);
                double x = r + rng.nextDouble() * (w - 2 * r);
                double y = r + rng.nextDouble() * (h - 2 * r);
                planets.add(randomPlanet(x, y));
            }
        }

        void clearTrails() {
            for (Planet p : planets) p.trail.clear();
        }

        private Planet randomPlanet(double x, double y) {
            double mass = cfg.minMass + rng.nextDouble() * (cfg.maxMass - cfg.minMass);
            double radius = cfg.minRadius + rng.nextDouble() * (cfg.maxRadius - cfg.minRadius);
            double angle = rng.nextDouble() * Math.PI * 2;
            double speed = rng.nextDouble() * cfg.maxInitialSpeed;
            Color color = Color.getHSBColor(rng.nextFloat(), 0.65f + rng.nextFloat() * 0.35f, 0.95f);
            return new Planet(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed, mass, radius, color);
        }

        // ----------------- Physics -----------------
        private void step(double dt) {
            int w = getWidth(), h = getHeight();
            if (w <= 0 || h <= 0) return;

            // 1) Pairwise Newtonian gravity with Plummer softening.
            for (Planet p : planets) { p.ax = 0; p.ay = 0; }
            List<Planet> list = new ArrayList<>(planets);
            double eps2 = cfg.softening * cfg.softening;
            for (int i = 0; i < list.size(); i++) {
                Planet a = list.get(i);
                for (int j = i + 1; j < list.size(); j++) {
                    Planet b = list.get(j);
                    double dx = b.x - a.x, dy = b.y - a.y;
                    double distSq = dx * dx + dy * dy + eps2;
                    double dist = Math.sqrt(distSq);
                    // F = G m1 m2 / r^2 ; acceleration a = F / m
                    double force = cfg.gravitationalConstant * a.mass * b.mass / distSq;
                    double fx = force * dx / dist, fy = force * dy / dist;
                    a.ax += fx / a.mass;  a.ay += fy / a.mass;
                    b.ax -= fx / b.mass;  b.ay -= fy / b.mass;
                }
            }

            // 2) Semi-implicit Euler: update velocity first, then position.
            for (Planet p : list) {
                p.vx += p.ax * dt;
                p.vy += p.ay * dt;
                p.x += p.vx * dt;
                p.y += p.vy * dt;
            }

            // 3) Planet-planet collisions (impulse-based, like billiard balls).
            //    Point masses have no cross-section, so there is nothing to resolve.
            if (cfg.planetCollisions && !cfg.pointMasses) {
                for (int i = 0; i < list.size(); i++) {
                    Planet a = list.get(i);
                    for (int j = i + 1; j < list.size(); j++) {
                        Planet b = list.get(j);
                        resolveCollision(a, b, cfg.planetRestitution);
                    }
                }
            }

            // 4) Inescapable box: reflect off the cushions. Restitution scales the
            //    whole velocity rather than just the reflected component, so every
            //    bounce costs speed — even a glancing one along the cushion.
            for (Planet p : list) {
                double r = p.radius;
                boolean bounced = false;
                if (p.x - r < 0)      { p.x = r;       p.vx = Math.abs(p.vx);  bounced = true; }
                else if (p.x + r > w) { p.x = w - r;   p.vx = -Math.abs(p.vx); bounced = true; }
                if (p.y - r < 0)      { p.y = r;       p.vy = Math.abs(p.vy);  bounced = true; }
                else if (p.y + r > h) { p.y = h - r;   p.vy = -Math.abs(p.vy); bounced = true; }
                if (bounced) {
                    p.vx *= cfg.wallRestitution;
                    p.vy *= cfg.wallRestitution;
                }
            }

            // 5) Trails.
            if (cfg.showTrails) {
                for (Planet p : list) {
                    p.trail.addLast(new Planet.Point2D(p.x, p.y));
                    while (p.trail.size() > cfg.trailLength) p.trail.removeFirst();
                }
            }
        }

        /** Impulse-based elastic/inelastic collision between two circles. */
        private void resolveCollision(Planet a, Planet b, double restitution) {
            double dx = b.x - a.x, dy = b.y - a.y;
            double dist = Math.sqrt(dx * dx + dy * dy);
            double minDist = a.radius + b.radius;
            if (dist >= minDist || dist == 0) return;

            double nx = dx / dist, ny = dy / dist;

            // Separate overlapping bodies proportionally to inverse mass.
            double overlap = minDist - dist;
            double invA = 1.0 / a.mass, invB = 1.0 / b.mass;
            double invSum = invA + invB;
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
                paintFlat(g2);
            }

            // Box edge + HUD
            g2.setColor(new Color(90, 110, 200));
            g2.setStroke(new BasicStroke(3));
            g2.drawRect(1, 1, getWidth() - 3, getHeight() - 3);
            g2.setStroke(new BasicStroke(1));
            g2.setColor(new Color(200, 210, 255, 180));
            g2.drawString(String.format("Planets: %d   %s   (click to add a planet)%s",
                    planets.size(), cfg.paused ? "PAUSED" : "running",
                    cfg.showGravityWell ? "   [gravitational well]" : ""), 12, 20);
        }

        /** The original top-down 2D view. */
        private void paintFlat(Graphics2D g2) {
            // Trails
            if (cfg.showTrails) {
                for (Planet p : planets) {
                    Planet.Point2D prev = null;
                    int i = 0, n = p.trail.size();
                    for (Planet.Point2D pt : p.trail) {
                        if (prev != null) {
                            float alpha = (float) i / n * 0.55f;
                            g2.setColor(new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(),
                                    (int) (alpha * 255)));
                            g2.drawLine((int) prev.x(), (int) prev.y(), (int) pt.x(), (int) pt.y());
                        }
                        prev = pt; i++;
                    }
                }
            }

            // Planets
            for (Planet p : planets) {
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

                // velocity vector
                if (cfg.showVectors) {
                    g2.setColor(Color.WHITE);
                    g2.drawLine((int) p.x, (int) p.y,
                            (int) (p.x + p.vx * 0.3), (int) (p.y + p.vy * 0.3));
                }
            }

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

        private double[][] wellZ;                 // sheet height, [cols+1][rows+1]
        private int wellCols, wellRows;
        private double wellCellW, wellCellH;

        /** Maximum dip, in world units, that the sheet is allowed to reach. */
        private double maxWellDepth(int h) { return h * 0.45; }

        /**
         * Squared softening length used when sampling a body's potential. Widened
         * beyond the body's own radius, and floored at roughly one mesh cell, so each
         * funnel spans several cells and reads as a bowl rather than a one-cell spike.
         */
        private double wellSoftening(Planet p) {
            double soft = Math.max(Math.max(p.radius * 2.0, cfg.softening), wellCellW * 0.8);
            return soft * soft;
        }

        /** Samples the gravitational potential onto the mesh grid. */
        private void computeWellGrid(int w, int h) {
            int cols = Math.max(8, cfg.wellResolution);
            int rows = Math.max(6, (int) Math.round(cols * (double) h / w));
            if (wellZ == null || wellCols != cols || wellRows != rows) {
                wellCols = cols; wellRows = rows;
                wellZ = new double[cols + 1][rows + 1];
            }
            wellCellW = (double) w / cols;
            wellCellH = (double) h / rows;

            double maxZ = maxWellDepth(h);
            double k = cfg.wellDepthScale;
            Planet[] ps = planets.toArray(new Planet[0]);
            double[] eps2 = new double[ps.length];
            for (int n = 0; n < ps.length; n++) eps2[n] = wellSoftening(ps[n]);

            for (int i = 0; i <= cols; i++) {
                double gx = i * wellCellW;
                for (int j = 0; j <= rows; j++) {
                    double gy = j * wellCellH;
                    double potential = 0;
                    for (int n = 0; n < ps.length; n++) {
                        double dx = gx - ps[n].x, dy = gy - ps[n].y;
                        potential += ps[n].mass / Math.sqrt(dx * dx + dy * dy + eps2[n]);
                    }
                    wellZ[i][j] = depthOf(potential, maxZ, k);
                }
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
        private double restingZ(Planet self, int h) {
            double potential = 0;
            for (Planet p : planets) {
                if (p == self) continue;
                double dx = self.x - p.x, dy = self.y - p.y;
                potential += p.mass / Math.sqrt(dx * dx + dy * dy + wellSoftening(p));
            }
            return depthOf(potential, maxWellDepth(h), cfg.wellDepthScale);
        }

        /** Bilinear lookup of the sheet height at an arbitrary point. */
        private double sampleWell(double x, double y) {
            if (wellZ == null) return 0;
            double fx = Math.max(0, Math.min(wellCols, x / wellCellW));
            double fy = Math.max(0, Math.min(wellRows, y / wellCellH));
            int i = Math.min((int) fx, wellCols - 1);
            int j = Math.min((int) fy, wellRows - 1);
            double tx = fx - i, ty = fy - j;
            double top = wellZ[i][j] * (1 - tx) + wellZ[i + 1][j] * tx;
            double bot = wellZ[i][j + 1] * (1 - tx) + wellZ[i + 1][j + 1] * tx;
            return top * (1 - ty) + bot * ty;
        }

        private void paintGravityWell(Graphics2D g2) {
            int w = getWidth(), h = getHeight();
            if (w <= 0 || h <= 0) return;

            computeWellGrid(w, h);
            WellView v = new WellView(w, h);
            double maxZ = maxWellDepth(h);

            // Bucket the planets by mesh row so they can be drawn interleaved with
            // the sheet — that gives correct occlusion without a depth buffer.
            List<List<Planet>> byRow = new ArrayList<>(wellRows);
            for (int j = 0; j < wellRows; j++) byRow.add(null);
            for (Planet p : planets) {
                int j = Math.max(0, Math.min(wellRows - 1, (int) (p.y / wellCellH)));
                if (byRow.get(j) == null) byRow.set(j, new ArrayList<>(2));
                byRow.get(j).add(p);
            }

            // Painter's algorithm: rows always run far -> near (cos(yaw) > 0);
            // columns run far -> near depending on the sign of sin(yaw).
            boolean colsAscending = v.sinY >= 0;
            int[] xs = new int[4], ys = new int[4];

            for (int j = 0; j < wellRows; j++) {
                double y0 = j * wellCellH, y1 = (j + 1) * wellCellH;
                for (int c = 0; c < wellCols; c++) {
                    int i = colsAscending ? c : wellCols - 1 - c;
                    double x0 = i * wellCellW, x1 = (i + 1) * wellCellW;
                    double z00 = wellZ[i][j], z10 = wellZ[i + 1][j];
                    double z01 = wellZ[i][j + 1], z11 = wellZ[i + 1][j + 1];

                    xs[0] = (int) v.sx(x0, y0); ys[0] = (int) v.sy(x0, y0, z00);
                    xs[1] = (int) v.sx(x1, y0); ys[1] = (int) v.sy(x1, y0, z10);
                    xs[2] = (int) v.sx(x1, y1); ys[2] = (int) v.sy(x1, y1, z11);
                    xs[3] = (int) v.sx(x0, y1); ys[3] = (int) v.sy(x0, y1, z01);

                    double avgZ = (z00 + z10 + z01 + z11) * 0.25;
                    double depth = Math.min(1, -avgZ / maxZ);
                    // Diffuse shading from the sheet's own slope.
                    double dzdx = ((z10 + z11) - (z00 + z01)) * 0.5 / wellCellW;
                    double dzdy = ((z01 + z11) - (z00 + z10)) * 0.5 / wellCellH;
                    double len = Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1);
                    double lambert = (0.45 * dzdx + 0.62 * dzdy + 0.65) / len;
                    double shade = 0.35 + 0.75 * Math.max(0, lambert);

                    g2.setColor(wellColor(depth, shade, false));
                    g2.fillPolygon(xs, ys, 4);
                    // Only the top and left edges: the neighbouring cells supply the
                    // rest, and the outer two edges are covered by the rim pass.
                    g2.setColor(wellColor(depth, shade, true));
                    g2.drawLine(xs[0], ys[0], xs[1], ys[1]);
                    g2.drawLine(xs[0], ys[0], xs[3], ys[3]);
                }

                List<Planet> here = byRow.get(j);
                if (here != null) {
                    if (here.size() > 1) {
                        here.sort(colsAscending ? Comparator.comparingDouble(p -> p.x)
                                                : Comparator.comparingDouble(p -> -p.x));
                    }
                    for (Planet p : here) drawPlanet3D(g2, v, p);
                }
            }

            // Rim of the box, riding the edge of the sheet.
            g2.setColor(new Color(120, 145, 235, 200));
            g2.setStroke(new BasicStroke(2f));
            drawWellEdge(g2, v, 0, 0, 1, 0);
            drawWellEdge(g2, v, 0, wellRows, 1, 0);
            drawWellEdge(g2, v, 0, 0, 0, 1);
            drawWellEdge(g2, v, wellCols, 0, 0, 1);
            g2.setStroke(new BasicStroke(1f));
        }

        /** Traces one edge of the mesh, following the sheet's height. */
        private void drawWellEdge(Graphics2D g2, WellView v, int i0, int j0, int di, int dj) {
            int steps = di != 0 ? wellCols : wellRows;
            int px = 0, py = 0;
            for (int s = 0; s <= steps; s++) {
                int i = i0 + di * s, j = j0 + dj * s;
                double x = i * wellCellW, y = j * wellCellH;
                int sx = (int) v.sx(x, y), sy = (int) v.sy(x, y, wellZ[i][j]);
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
            double z = restingZ(p, v.h) + radius;
            double cx = v.sx(p.x, p.y);
            double cy = v.sy(p.x, p.y, z);
            double r = radius * v.scale;

            // Trail, projected down onto the sheet.
            if (cfg.showTrails && !p.trail.isEmpty()) {
                Planet.Point2D prev = null;
                int i = 0, n = p.trail.size();
                for (Planet.Point2D pt : p.trail) {
                    if (prev != null) {
                        float alpha = (float) i / n * 0.6f;
                        g2.setColor(new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(),
                                (int) (alpha * 255)));
                        g2.drawLine((int) v.sx(prev.x(), prev.y()),
                                (int) v.sy(prev.x(), prev.y(), sampleWell(prev.x(), prev.y())),
                                (int) v.sx(pt.x(), pt.y()),
                                (int) v.sy(pt.x(), pt.y(), sampleWell(pt.x(), pt.y())));
                    }
                    prev = pt; i++;
                }
            }

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
            final int w, h;
            final double cosP, sinP, cosY, sinY, scale, cx, cy;

            WellView(int w, int h) {
                this.w = w; this.h = h;
                double pitch = Math.toRadians(cfg.wellPitch);
                double yaw = Math.toRadians(cfg.wellYaw);
                cosP = Math.cos(pitch); sinP = Math.sin(pitch);
                cosY = Math.cos(yaw);   sinY = Math.sin(yaw);
                // Shrink so the rotated footprint still fits across the canvas.
                double footprint = Math.abs(w * cosY) + Math.abs(h * sinY);
                scale = 0.92 * w / footprint;
                cx = w / 2.0;
                cy = h * 0.38;   // plane sits high; the wells hang below it
            }

            /** Screen x of a plane point. */
            double sx(double x, double y) {
                return cx + ((x - w / 2.0) * cosY - (y - h / 2.0) * sinY) * scale;
            }

            /** Screen y of a plane point lifted to height z. */
            double sy(double x, double y, double z) {
                double depth = (x - w / 2.0) * sinY + (y - h / 2.0) * cosY;
                return cy + (depth * cosP - z * sinP) * scale;
            }

            /** Screen point back onto the z = 0 plane (used for click-to-add). */
            double[] unproject(double screenX, double screenY) {
                double xr = (screenX - cx) / scale;
                double yr = (screenY - cy) / (scale * Math.max(1e-6, cosP));
                return new double[]{
                        xr * cosY + yr * sinY + w / 2.0,
                        -xr * sinY + yr * cosY + h / 2.0};
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
            JPanel spawn = group("Spawn Settings (applied on Reset)");
            spawn.add(slider("Planet count", 1, 40, cfg.planetCount,
                    v -> cfg.planetCount = (int) v, "%.0f", 1));
            spawn.add(slider("Min mass", 10, 2000, (int) cfg.minMass,
                    v -> cfg.minMass = Math.min(v, cfg.maxMass), "%.0f", 1));
            spawn.add(slider("Max mass", 10, 5000, (int) cfg.maxMass,
                    v -> cfg.maxMass = Math.max(v, cfg.minMass), "%.0f", 1));
            spawn.add(slider("Min radius", 2, 30, (int) cfg.minRadius,
                    v -> cfg.minRadius = Math.min(v, cfg.maxRadius), "%.0f", 1));
            spawn.add(slider("Max radius", 2, 50, (int) cfg.maxRadius,
                    v -> cfg.maxRadius = Math.max(v, cfg.minRadius), "%.0f", 1));
            spawn.add(slider("Max initial speed", 0, 300, (int) cfg.maxInitialSpeed,
                    v -> cfg.maxInitialSpeed = v, "%.0f", 1));
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

            JButton resetBtn = new JButton("Reset (respawn planets)");
            resetBtn.addActionListener(e -> sim.respawnPlanets());
            buttons.add(resetBtn);

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