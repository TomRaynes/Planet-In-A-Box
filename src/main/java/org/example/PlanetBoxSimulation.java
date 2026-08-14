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
 *   - Wall restitution (bounciness, 1.0 = perfectly elastic)
 *   - Planet-planet collisions on/off, with their own restitution
 *   - Gravity softening (prevents singular forces at tiny separations)
 *   - Number of planets, mass range and radius range used on reset
 *   - Trails on/off and trail length
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
        volatile double wallRestitution = 1.0;          // 1 = elastic cushion
        volatile double planetRestitution = 0.9;        // for planet-planet impacts
        volatile boolean planetCollisions = true;
        volatile double softening = 4.0;                // gravity softening length (px)
        volatile boolean showTrails = true;
        volatile int trailLength = 200;
        volatile boolean showVectors = false;
        volatile boolean paused = false;

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
                    planets.add(randomPlanet(e.getX(), e.getY()));
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
            if (cfg.planetCollisions) {
                for (int i = 0; i < list.size(); i++) {
                    Planet a = list.get(i);
                    for (int j = i + 1; j < list.size(); j++) {
                        Planet b = list.get(j);
                        resolveCollision(a, b, cfg.planetRestitution);
                    }
                }
            }

            // 4) Inescapable box: reflect off the cushions.
            for (Planet p : list) {
                if (p.x - p.radius < 0)      { p.x = p.radius;       p.vx = Math.abs(p.vx) * cfg.wallRestitution; }
                else if (p.x + p.radius > w) { p.x = w - p.radius;   p.vx = -Math.abs(p.vx) * cfg.wallRestitution; }
                if (p.y - p.radius < 0)      { p.y = p.radius;       p.vy = Math.abs(p.vy) * cfg.wallRestitution; }
                else if (p.y + p.radius > h) { p.y = h - p.radius;   p.vy = -Math.abs(p.vy) * cfg.wallRestitution; }
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
                int d = (int) (p.radius * 2);
                // soft glow
                g2.setColor(new Color(p.color.getRed(), p.color.getGreen(), p.color.getBlue(), 40));
                g2.fillOval((int) (p.x - p.radius * 1.8), (int) (p.y - p.radius * 1.8),
                        (int) (d * 1.8), (int) (d * 1.8));
                g2.setColor(p.color);
                g2.fillOval((int) (p.x - p.radius), (int) (p.y - p.radius), d, d);
                g2.setColor(p.color.brighter());
                g2.drawOval((int) (p.x - p.radius), (int) (p.y - p.radius), d, d);

                // velocity vector
                if (cfg.showVectors) {
                    g2.setColor(Color.WHITE);
                    g2.drawLine((int) p.x, (int) p.y,
                            (int) (p.x + p.vx * 0.3), (int) (p.y + p.vy * 0.3));
                }
            }

            // Box edge + HUD
            g2.setColor(new Color(90, 110, 200));
            g2.setStroke(new BasicStroke(3));
            g2.drawRect(1, 1, getWidth() - 3, getHeight() - 3);
            g2.setStroke(new BasicStroke(1));
            g2.setColor(new Color(200, 210, 255, 180));
            g2.drawString(String.format("Planets: %d   %s   (click to add a planet)",
                    planets.size(), cfg.paused ? "PAUSED" : "running"), 12, 20);
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
            collisions.add(slider("Wall bounciness (%)", 0, 100, (int) (cfg.wallRestitution * 100),
                    v -> cfg.wallRestitution = v / 100.0, "%.2f", 0.01));
            JCheckBox collideBox = new JCheckBox("Planet-planet collisions", cfg.planetCollisions);
            collideBox.addActionListener(e -> cfg.planetCollisions = collideBox.isSelected());
            collideBox.setAlignmentX(LEFT_ALIGNMENT);
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