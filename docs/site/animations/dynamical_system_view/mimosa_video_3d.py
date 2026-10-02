"""3D captioned Manim video: the dynamical-systems view of LLM agents.

Nine ThreeDScene scenes. An agent is a flow on a potential landscape;
a multi-agent workflow is a hybrid system: flows chained by discrete,
lossy jump maps. No LaTeX; all math is Pango/Unicode text.
"""
import numpy as np
from manim import *

CAPTION_SIZE = 26
CAPTION_COLOR = GREY_A
HEADER_COLOR = YELLOW_B
GOOD = GREEN_C
BAD = RED_C
KEY = ORANGE
MAX_CAPTION_WIDTH = 12.8
SURF_RES = (26, 26)
Z_OFFSET = 0.13
CAP_PACE, CAP_FLOOR = 1.75, 1.5


# ----------------------------------------------------------- landscape maths
def gaussian_well(x, y, cx, cy, depth, width):
    """A single attracting well centered at (cx, cy)."""
    return -depth * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / width)


def main_potential(x, y, deepen=1.0):
    """Three-well landscape; `deepen` scales well A (lock-in visual)."""
    bowl = 0.065 * (x * x + y * y)
    return (bowl
            + gaussian_well(x, y, -1.7, -0.9, 1.05 * deepen, 1.3)
            + gaussian_well(x, y, 1.6, 1.0, 0.9, 1.2)
            + gaussian_well(x, y, 0.9, -1.9, 0.75, 1.0))


def duo_potential(x, y):
    """Two wells with a saddle at the origin (placement scenes)."""
    bowl = 0.09 * (x * x + y * y)
    return (bowl
            + gaussian_well(x, y, -1.6, 0, 1.0, 1.1)
            + gaussian_well(x, y, 1.6, 0, 1.0, 1.1))


def grad_of(potential, p, h=1e-3):
    """Numerical gradient of a 2-D potential at point p."""
    return np.array([
        (potential(p[0] + h, p[1]) - potential(p[0] - h, p[1])) / (2 * h),
        (potential(p[0], p[1] + h) - potential(p[0], p[1] - h)) / (2 * h),
    ])


def descend_path(potential, start, steps=320, dt=0.022, swirl=0.85):
    """Gradient-descent trajectory with a tangential swirl (an orbit-like fall)."""
    p, pts = np.array(start, float), []
    for _ in range(steps):
        g = grad_of(potential, p)
        v = -g + swirl * np.array([-g[1], g[0]])
        speed = np.linalg.norm(v)
        v = v / max(speed, 1e-6) * min(speed, 3.0)
        p = p + dt * v
        pts.append(p.copy())
    return pts


def lift(potential, xy_points, x_shift=0.0, every=6):
    """Lift 2-D points onto the surface (subsampled), slightly above it."""
    sub = xy_points[::every] + [xy_points[-1]]
    return [np.array([x + x_shift, y, potential(x, y) + Z_OFFSET]) for x, y in sub]


def smooth_3d(points, color, width=4):
    """A smooth VMobject through 3-D points."""
    path = VMobject(stroke_color=color, stroke_width=width)
    path.set_points_smoothly(points)
    return path


def build_surface(potential, span, colors, x_shift=0.0, deepen=1.0):
    """The landscape as a checkerboard Surface over [-span, span]^2."""
    def point(u, v):
        return np.array([u + x_shift, v, potential(u, v) if deepen == 1.0
                         else potential(u, v, deepen)])
    return Surface(point, u_range=[-span, span], v_range=[-span, span],
                   resolution=SURF_RES, fill_opacity=0.75,
                   checkerboard_colors=colors, stroke_color=GREY_D,
                   stroke_width=0.4)


def cut_plane(x0, color, y_span=4.6, z_span=2.4):
    """A translucent vertical plane at x = x0 (a candidate boundary)."""
    plane = Rectangle(width=z_span, height=y_span)
    plane.set_fill(color, opacity=0.22).set_stroke(color, width=2.5)
    plane.rotate(PI / 2, axis=UP).move_to([x0, 0, -0.15])
    return plane


# ----------------------------------------------------------------- base scene
class Mimosa3D(ThreeDScene):
    """ThreeDScene with screen-fixed crossfading captions and headers."""

    def setup(self):
        self._caption = None

    def cap(self, text, wait=3.0):
        """Show a fixed-in-frame caption, crossfading from the previous one."""
        new = Text(text, font_size=CAPTION_SIZE, color=CAPTION_COLOR)
        if new.width > MAX_CAPTION_WIDTH:
            new.scale_to_fit_width(MAX_CAPTION_WIDTH)
        new.to_edge(DOWN, buff=0.32)
        self.add_fixed_in_frame_mobjects(new)
        anims = [FadeIn(new, shift=UP * 0.2)]
        if self._caption is not None:
            anims.append(FadeOut(self._caption, shift=UP * 0.2))
        self.play(*anims, run_time=0.6)
        self._caption = new
        self.wait(wait * CAP_PACE + CAP_FLOOR)

    def clear_cap(self):
        if self._caption is not None:
            self.play(FadeOut(self._caption), run_time=0.4)
            self._caption = None

    def header(self, text):
        h = Text(text, font_size=30, weight=BOLD, color=HEADER_COLOR)
        h.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(h)
        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.6)
        return h

    def fixed_formula(self, text, size=28, color=WHITE):
        f = Text(text, font_size=size, color=color, slant=ITALIC)
        self.add_fixed_in_frame_mobjects(f)
        return f


# ------------------------------------------------------------------ scene 01
class D01Title(Mimosa3D):
    def construct(self):
        self.show_title_card()
        self.show_state_flow()

    def show_title_card(self):
        title = Text("The Agent as a Dynamical System", font_size=42, weight=BOLD)
        sub = Text("flows, basins, and chained systems — the Mimosa framework in 3D",
                   font_size=26, color=GREY_A)
        card = VGroup(title, sub).arrange(DOWN, buff=0.45)
        self.add_fixed_in_frame_mobjects(card)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.0)
        self.play(FadeIn(sub), run_time=0.7)
        self.wait(2.6)
        self.play(FadeOut(card), run_time=0.6)

    def show_state_flow(self):
        self.set_camera_orientation(phi=64 * DEGREES, theta=-48 * DEGREES, zoom=0.9)
        axes = ThreeDAxes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-2, 2, 1],
                          x_length=6, y_length=6, z_length=3.4)
        spiral = ParametricFunction(
            lambda t: np.array([2.3 * np.exp(-0.28 * t) * np.cos(2.1 * t),
                                2.3 * np.exp(-0.28 * t) * np.sin(2.1 * t),
                                1.6 * np.exp(-0.35 * t) - 0.2]),
            t_range=[0, 7], color=BLUE_C, stroke_width=4)
        dot = Dot3D(spiral.get_start(), radius=0.09, color=BLUE_C)
        self.play(Create(axes), run_time=1.2)
        self.add(dot)
        self.cap("Forget the transcript. An agent is a state evolving in a high-dimensional space.", wait=0.2)
        self.begin_ambient_camera_rotation(rate=0.07)
        self.play(Create(spiral), MoveAlongPath(dot, spiral, rate_func=linear), run_time=4.5)
        self.stop_ambient_camera_rotation()
        self.cap("Tokens are its shadow. The dynamics live up here — and dynamics is a language with theorems.", wait=3.0)


# ------------------------------------------------------------------ scene 02
class D02SystemMap(Mimosa3D):
    ORBIT_A = [-2.6, 1.9, 1.5]
    ORBIT_B = [-2.4, -2.1, 1.4]

    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-45 * DEGREES, zoom=0.9)
        self.header("1 · State, map, initial condition")
        eq = self.fixed_formula("x(t+1) = F( x(t), context )", size=30)
        eq.to_edge(UP, buff=1.1)
        self.play(Write(eq), run_time=1.0)
        self.cap("One update rule F — the frozen weights — applied over and over to a state.", wait=2.6)
        self.show_orbit(self.ORBIT_A, [1.5, 1.2, -0.6], TEAL_C)
        self.cap("The prompt is not an instruction. It is an initial condition.", wait=2.6)
        self.show_orbit(self.ORBIT_B, [1.2, -1.6, -0.5], PURPLE_C)
        self.cap("Same F, different initial condition — a different orbit, a different fate. Everything else follows from this.", wait=3.4)

    def show_orbit(self, start, end, color):
        """Discrete orbit: iterates hopping toward a fixed point."""
        start, end = np.array(start, float), np.array(end, float)
        pts = [start + (end - start) * (1 - 0.55 ** k)
               + 0.35 * (0.6 ** k) * np.array([np.sin(2.2 * k), np.cos(2.2 * k), 0])
               for k in range(8)]
        dots = VGroup(*[Dot3D(p, radius=0.07, color=color) for p in pts])
        hops = VGroup(*[Line(pts[i], pts[i + 1], color=color, stroke_width=3)
                        for i in range(len(pts) - 1)])
        anchor = Dot3D(end, radius=0.11, color=YELLOW)
        self.play(FadeIn(dots[0]), run_time=0.4)
        self.play(LaggedStart(*[AnimationGroup(Create(hops[i]), FadeIn(dots[i + 1]))
                                for i in range(len(pts) - 1)], lag_ratio=0.5),
                  run_time=2.6)
        self.play(FadeIn(anchor, scale=1.6), run_time=0.5)


# ------------------------------------------------------------------ scene 03
class D03Landscape(Mimosa3D):
    SPAN = 3.2

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-55 * DEGREES, zoom=0.8)
        self.header("2 · The landscape — basins are concepts")
        surface = build_surface(main_potential, self.SPAN, [BLUE_E, BLUE_D])
        self.play(Create(surface), run_time=2.2)
        self.cap("The state space is not flat: it is a landscape of basins — code, prose, math. One basin per concept.", wait=0.2)
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(3.2)
        self.stop_ambient_camera_rotation()
        self.roll_ball([-0.4, 2.6], BLUE_C, 3.0)
        self.cap("A prompt drops a state onto the landscape. Routing = falling into a basin.", wait=2.2)
        self.roll_ball([-0.2, 2.5], TEAL_C, 2.4)
        self.cap("Nearby initial conditions, same well: that is why paraphrases behave alike. Transverse contraction.", wait=2.8)
        self.roll_ball([0.35, 1.9], KEY, 2.6)
        self.cap("But near a ridge, a tiny nudge picks a different basin. Ridges are the key decisions — sparse, and sensitive.", wait=3.4)

    def roll_ball(self, start_xy, color, run_time):
        path = smooth_3d(lift(main_potential, descend_path(main_potential, start_xy)),
                         color)
        ball = Dot3D(path.get_start(), radius=0.09, color=color)
        self.add(ball)
        self.play(Create(path), MoveAlongPath(ball, path, rate_func=linear),
                  run_time=run_time)


# ------------------------------------------------------------------ scene 04
class D04WithinBasin(Mimosa3D):
    SPAN = 3.2

    def construct(self):
        self.set_camera_orientation(phi=58 * DEGREES, theta=-60 * DEGREES, zoom=0.85)
        self.header("3 · Inside a basin — lock-in and escape")
        self.surface = build_surface(main_potential, self.SPAN, [BLUE_E, BLUE_D])
        self.play(Create(self.surface), run_time=1.8)
        self.show_deepening()
        self.show_escape_attempt()

    def show_deepening(self):
        path = smooth_3d(lift(main_potential,
                              descend_path(main_potential, [-2.9, 0.9], steps=380)),
                         RED_C)
        ball = Dot3D(path.get_start(), radius=0.09, color=RED_C)
        self.add(ball)
        self.cap("Suppose the state falls into the WRONG well — one bad key decision.", wait=0.2)
        self.play(Create(path), MoveAlongPath(ball, path, rate_func=linear), run_time=3.0)
        deeper = build_surface(main_potential, self.SPAN, [BLUE_E, BLUE_D], deepen=1.9)
        self.cap("Now generation conditions on its own output — and the occupied well deepens. That is lock-in, as geometry.", wait=0.2)
        self.play(Transform(self.surface, deeper), run_time=2.4)
        self.wait(1.6)
        self.ball = ball

    def show_escape_attempt(self):
        eq = self.fixed_formula("P(escape)  ~  ρ^t ,   ρ < 1", size=28, color=RED_C)
        eq.to_edge(RIGHT, buff=0.6).shift(UP * 2.2)
        rim_try = smooth_3d([self.ball.get_center(),
                             self.ball.get_center() + np.array([0.5, 0.55, 0.5]),
                             self.ball.get_center() + np.array([0.15, 0.2, 0.1])],
                            RED_C, width=3)
        self.play(Create(rim_try), Write(eq), run_time=1.4)
        self.cap("Perturbations climb the wall and fall back: escape probability decays geometrically with every step.", wait=3.0)
        self.cap("Coherent, fluent, wrong — and trapped. Terminal collapse is not error pile-up; it is capture.", wait=3.2)


# ------------------------------------------------------------------ scene 05
class D05Hallucination(Mimosa3D):
    SPAN = 3.2

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-40 * DEGREES, zoom=0.85)
        self.header("4 · Itinerancy — the uncontrolled jump")
        surface = build_surface(main_potential, self.SPAN, [BLUE_E, BLUE_D])
        self.play(Create(surface), run_time=1.8)
        self.show_itinerant_run()

    def show_itinerant_run(self):
        settle = descend_path(main_potential, [1.2, 2.6], steps=260)
        exit_hop = [np.array(settle[-1]) + s * (np.array([0.9, -1.9]) - settle[-1])
                    for s in np.linspace(0, 1, 30)]
        recapture = descend_path(main_potential, exit_hop[-1], steps=200)
        blue = smooth_3d(lift(main_potential, settle), BLUE_C)
        hop = smooth_3d(lift(main_potential, exit_hop, every=3), KEY, width=5)
        red = smooth_3d(lift(main_potential, recapture), RED_C)
        ball = Dot3D(blue.get_start(), radius=0.09, color=BLUE_C)
        self.add(ball)
        self.cap("Along the flow, the dynamics are expansive: small wobbles grow while the walls hold — until they don't.", wait=0.2)
        self.play(Create(blue), MoveAlongPath(ball, blue, rate_func=linear), run_time=2.8)
        ball.set_color(KEY)
        self.play(Create(hop), MoveAlongPath(ball, hop, rate_func=linear), run_time=1.4)
        ball.set_color(RED_C)
        self.play(Create(red), MoveAlongPath(ball, red, rate_func=linear), run_time=2.2)
        self.cap("An unplanned basin exit, then confident contraction into the wrong well. That is a hallucination — in one picture.", wait=3.4)
        self.cap("Left alone, the system itinerates: it WILL jump between basins. The only question is whether anyone chooses where.", wait=3.4)


# ------------------------------------------------------------------ scene 06
class D06ChainedSystems(Mimosa3D):
    """The centerpiece: two landscapes chained by a lossy 1-D jump."""
    SPAN = 1.9
    SHIFT = 2.95
    RAIL_Z = -1.5

    def construct(self):
        self.set_camera_orientation(phi=62 * DEGREES, theta=-90 * DEGREES, zoom=1.0)
        self.header("5 · A workflow is a CHAIN of dynamical systems")
        self.build_two_landscapes()
        self.flow_in_system_a()
        self.jump_across()
        self.flow_in_system_b()
        self.state_the_hybrid_view()

    def build_two_landscapes(self):
        self.surf_a = build_surface(duo_potential, self.SPAN, [BLUE_E, BLUE_D],
                                    x_shift=-self.SHIFT)
        self.surf_b = build_surface(duo_potential, self.SPAN, [TEAL_E, TEAL_D],
                                    x_shift=self.SHIFT)
        tag_a = self.fixed_formula("system A", size=24, color=BLUE_B).move_to([-3.9, 2.45, 0])
        tag_b = self.fixed_formula("system B", size=24, color=TEAL_B).move_to([3.9, 2.45, 0])
        self.play(Create(self.surf_a), Create(self.surf_b),
                  FadeIn(tag_a), FadeIn(tag_b), run_time=2.4)
        self.cap("Agent A and agent B are two flows — each with its own landscape, its own basins, its own coherence pressure.", wait=2.8)

    def flow_in_system_a(self):
        xy = descend_path(duo_potential, [-1.2, 1.6], steps=300)
        path = smooth_3d(lift(duo_potential, xy, x_shift=-self.SHIFT), BLUE_C)
        self.ball_a = Dot3D(path.get_start(), radius=0.09, color=BLUE_C)
        self.end_a = path.get_end()
        self.path_a = path
        self.add(self.ball_a)
        self.play(Create(path), MoveAlongPath(self.ball_a, path, rate_func=linear),
                  run_time=2.8)
        self.cap("A runs: routes, contracts, accumulates a whole trajectory of state — curvature, alternatives, context.", wait=2.6)

    def jump_across(self):
        rail = Line([-1.1, 0, self.RAIL_Z], [1.1, 0, self.RAIL_Z],
                    color=GREY_A, stroke_width=5)
        rail_tag = self.fixed_formula("tokens: a 1-D projection", size=22, color=GREY_A)
        rail_tag.move_to([0, -2.6, 0])
        drop = DashedLine(self.end_a, rail.get_start(), color=KEY, dash_length=0.12)
        msg = Dot3D(rail.get_start(), radius=0.07, color=KEY)
        self.play(Create(rail), FadeIn(rail_tag), Create(drop), run_time=1.4)
        self.cap("Then the handoff: the full 3-D state collapses onto a line. φ keeps a projection — nothing more.", wait=0.2)
        self.play(FadeOut(self.path_a), FadeOut(self.ball_a),
                  FadeIn(msg), run_time=1.2)
        self.cap("Watch what left the picture: the trajectory is GONE. Its residue cannot ride the rail.", wait=2.8)
        self.play(MoveAlongPath(msg, rail, rate_func=linear), run_time=1.4)
        self.msg = msg

    def flow_in_system_b(self):
        entry_xy = [-1.5, 1.3]
        entry = np.array([entry_xy[0] + self.SHIFT, entry_xy[1],
                          duo_potential(*entry_xy) + Z_OFFSET])
        lift_line = DashedLine(self.msg.get_center(), entry, color=KEY, dash_length=0.12)
        xy = descend_path(duo_potential, entry_xy, steps=300)
        path = smooth_3d(lift(duo_potential, xy, x_shift=self.SHIFT), TEAL_C)
        ball = Dot3D(entry, radius=0.09, color=TEAL_C)
        self.play(Create(lift_line), FadeIn(ball), FadeOut(self.msg), run_time=1.2)
        self.cap("ι re-injects it as B's initial condition. B must re-contract from scratch — and can misroute doing it.", wait=0.2)
        self.play(Create(path), MoveAlongPath(ball, path, rate_func=linear), run_time=2.6)
        self.cap("But look what B gets: a FRESH landscape. No deepened well, no inherited coherence pressure, independent errors.", wait=3.2)

    def state_the_hybrid_view(self):
        eq = self.fixed_formula("flow A   →   x_B(0) = ι(φ(x_A))   →   flow B", size=28,
                                color=YELLOW_C)
        eq.to_edge(UP, buff=1.1)
        self.play(Write(eq), run_time=1.2)
        self.cap("This is a hybrid dynamical system: continuous flows, punctuated by discrete, lossy jump maps.", wait=3.0)
        self.cap("Every multi-agent framework — whatever its diagram — is exactly this object. Design = choosing the jumps.", wait=3.4)


# ------------------------------------------------------------------ scene 07
class D07Placement(Mimosa3D):
    SPAN = 3.0
    WAYPOINTS = [(-2.8, 1.7), (-2.1, 0.8), (-1.6, 0.15), (-1.1, -0.4),
                 (-0.5, -0.2), (0.0, 0.0), (0.6, 0.25), (1.2, 0.45),
                 (1.75, 0.1), (1.45, -0.3), (1.6, 0.05)]

    def construct(self):
        self.set_camera_orientation(phi=58 * DEGREES, theta=-75 * DEGREES, zoom=0.8)
        self.header("6 · Where to cut the chain")
        surface = build_surface(duo_potential, self.SPAN, [BLUE_E, BLUE_D])
        self.play(Create(surface), run_time=1.8)
        self.ride_over_saddle()
        self.compare_cuts()

    def ride_over_saddle(self):
        pts = lift(duo_potential, [np.array(w) for w in self.WAYPOINTS], every=1)
        path = smooth_3d(pts, BLUE_C)
        ball = Dot3D(path.get_start(), radius=0.09, color=BLUE_C)
        self.add(ball)
        self.cap("Run the task as ONE flow and watch: it rides well 1, then crosses the saddle — the transition the task itself demands.", wait=0.2)
        self.play(Create(path), MoveAlongPath(ball, path, rate_func=linear), run_time=4.0)
        self.play(Flash(np.array([0, 0, duo_potential(0, 0) + 0.1]), color=KEY),
                  run_time=0.8)
        self.cap("At the saddle the state is between basins: maximum entropy, minimum committed structure.", wait=2.8)

    def compare_cuts(self):
        good = cut_plane(0.0, GOOD)
        good_tag = self.fixed_formula("cut at the saddle:  ΔJ > 0  ✓", size=24, color=GOOD)
        good_tag.move_to([3.4, 2.4, 0])
        bad = cut_plane(-1.6, BAD)
        bad_tag = self.fixed_formula("cut mid-well:  ΔJ < 0  ✗", size=24, color=BAD)
        bad_tag.move_to([-3.6, 2.4, 0])
        self.play(FadeIn(good), FadeIn(good_tag), run_time=1.0)
        self.cap("A jump placed there costs nothing it wasn't already paying — and resets everything worth resetting.", wait=2.8)
        self.play(FadeIn(bad), FadeIn(bad_tag), run_time=1.0)
        self.cap("A jump mid-well shatters live state, freezes half-formed beliefs, and forces a re-route the task never asked for.", wait=3.0)
        tagline = Text("Cut where the dynamics already want to jump.",
                       font_size=32, weight=BOLD, color=YELLOW_B).to_edge(UP, buff=1.15)
        self.add_fixed_in_frame_mobjects(tagline)
        self.play(FadeIn(tagline, scale=1.08), run_time=1.0)
        self.wait(2.4)


# ------------------------------------------------------------------ scene 08
class D08Evolution(Mimosa3D):
    SPAN = 3.0
    SAMPLES = [(-2.3, 0.34), (1.9, 0.41), (-0.9, 0.55), (0.8, 0.62),
               (-0.35, 0.74), (0.3, 0.79), (0.05, 0.86)]

    def construct(self):
        self.set_camera_orientation(phi=58 * DEGREES, theta=-75 * DEGREES, zoom=0.8)
        self.header("7 · Evolution — searching for the saddle, blind")
        surface = build_surface(duo_potential, self.SPAN, [BLUE_E, BLUE_D])
        self.play(Create(surface), run_time=1.8)
        self.cap("One catch: the landscape is invisible. It is model-specific, and no one gets to see it before running the system.", wait=3.0)
        self.sweep_and_score()

    def sweep_and_score(self):
        slider = ValueTracker(self.SAMPLES[0][0])
        plane = always_redraw(lambda: cut_plane(slider.get_value(), YELLOW_C))
        self.add(plane)
        axes, dots = self.fitness_panel()
        self.cap("So evolution samples: place a jump, run the whole chain, score the outcome. A black-box query of the landscape.", wait=2.0)
        for (x, fit), dot in zip(self.SAMPLES, dots):
            self.play(slider.animate.set_value(x), run_time=0.7)
            self.play(FadeIn(dot, scale=1.6), run_time=0.3)
        marker = DashedLine(axes.c2p(0.05, 0), axes.c2p(0.05, 1),
                            color=GOOD, dash_length=0.08)
        self.add_fixed_in_frame_mobjects(marker)
        self.play(Create(marker), run_time=0.8)
        self.cap("Selection keeps the cuts that score — and they pile up at the saddle. The population has FOUND the geometry.", wait=3.2)
        self.cap("Evolution here is not prompt tuning. It is system identification of an invisible landscape, from rollouts alone.", wait=3.6)

    def fitness_panel(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1, 0.5],
                    x_length=4.0, y_length=1.9,
                    axis_config={"include_tip": False, "font_size": 14})
        axes.to_corner(UR, buff=0.4).shift(DOWN * 1.35)
        label = Text("fitness vs cut position", font_size=18, color=GREY_A)
        label.next_to(axes, DOWN, buff=0.15)
        dots = [Dot(axes.c2p(x, f), radius=0.05, color=YELLOW_C)
                for x, f in self.SAMPLES]
        self.add_fixed_in_frame_mobjects(axes, label, *dots)
        self.remove(*dots)
        self.play(Create(axes), FadeIn(label), run_time=1.0)
        return axes, dots


# ------------------------------------------------------------------ scene 09
class D09End(Mimosa3D):
    LINES = (
        "one agent  =  a flow on an invisible landscape",
        "failure  =  capture by the wrong basin, walls deepening",
        "a workflow  =  flows chained by discrete, lossy jumps",
        "design  =  placing jumps where the flow crosses saddles",
        "evolution  =  identifying the landscape from rollouts",
    )

    def construct(self):
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        items = VGroup(*[Text(t, font_size=26) for t in self.LINES])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(UP * 0.4)
        self.add_fixed_in_frame_mobjects(items)
        self.remove(items)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in items],
                              lag_ratio=0.45), run_time=4.0)
        self.wait(4.0)
        self.play(FadeOut(items), run_time=0.8)
        tag = Text("A workflow is a controlled-itinerancy schedule.",
                   font_size=36, weight=BOLD, color=YELLOW_B)
        brand = Text("Mimosa · the dynamical-systems view", font_size=24, color=GREY_A)
        card = VGroup(tag, brand).arrange(DOWN, buff=0.6)
        self.add_fixed_in_frame_mobjects(card)
        self.remove(card)
        self.play(FadeIn(tag, scale=1.08), run_time=1.2)
        self.play(FadeIn(brand), run_time=0.8)
        self.wait(3.2)
        self.play(FadeOut(card), run_time=1.0)


SCENES = [D01Title, D02SystemMap, D03Landscape, D04WithinBasin, D05Hallucination,
          D06ChainedSystems, D07Placement, D08Evolution, D09End]

if __name__ == "__main__":
    p = descend_path(main_potential, [-0.4, 2.6], steps=50)
    assert len(p) == 50 and np.isfinite(p[-1]).all(), "descent integrator broken"
    print(f"{len(SCENES)} scenes defined; descent smoke check OK, end={p[-1].round(2)}")
