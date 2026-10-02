from manim import *
import numpy as np

# Color palette
C = {
    "bg": "#0d1117",
    "text": "#e6edf3",
    "agent_a": "#4a9eff",
    "agent_b": "#2ed573",
    "agent_c": "#ff6b35",
    "critic": "#ff6b9d",
    "message": "#a55eea",
    "attractor": "#ff6b35",
    "error": "#ff4757",
    "good": "#00ff88",
    "control": "#ffa502",
    "fast": "#4a9eff",
    "slow": "#ff6b9d",
    "grey": "#888888",
    "full": "#8b949e",
}


def make_contours(center, base_color, n=6, r_max=1.6, squash=0.65, seed=0):
    group = VGroup()
    for i, r in enumerate(np.linspace(0.25, r_max, n)):
        contour = VMobject(color=base_color, stroke_width=2, stroke_opacity=0.75 - i * 0.09)
        pts = []
        for t in np.linspace(0, 2 * PI, 60):
            noise = 0.06 * np.sin(4 * t + i + seed)
            x = center[0] + (r + noise) * np.cos(t)
            y = center[1] + (r + noise) * squash * np.sin(t)
            pts.append(np.array([x, y, 0]))
        contour.set_points_smoothly(pts + [pts[0]])
        group.add(contour)
    return group


def potential(x, a):
    """Tilted double well: left well annihilated as a grows (saddle-node at a≈0.385)."""
    return x**4 / 4 - x**2 / 2 - a * x


def left_min(a):
    """Left local minimum of the potential, or None past the bifurcation."""
    roots = np.roots([1, 0, -1, -a])
    real = [r.real for r in roots if abs(r.imag) < 1e-9]
    cand = [r for r in real if r < -1 / np.sqrt(3)]
    return min(cand) if cand else None


def right_min(a):
    roots = np.roots([1, 0, -1, -a])
    real = [r.real for r in roots if abs(r.imag) < 1e-9]
    cand = [r for r in real if r > 1 / np.sqrt(3)]
    return max(cand) if cand else None


class ControlTheoreticMASv2(Scene):
    """
    Patched presentation: state-space duality made explicit, parametric view
    as a stated modeling choice, drift as a slow-fast system with bifurcation.
    """

    def construct(self):
        self.camera.background_color = C["bg"]

        self.scene_title()
        self.scene_state_space_duality()
        self.scene_parametric_coupling()
        self.scene_slow_fast_drift()
        self.scene_network_of_systems()
        self.scene_control_mapping()
        self.scene_open_vs_closed_loop()
        self.scene_small_gain()
        self.scene_observability()
        self.scene_two_timescales()
        self.scene_summary()
        self.wait(1)

    # ------------------------------------------------------------------
    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)

    # ------------------------------------------------------------------
    def scene_title(self):
        title = Text(
            "Multi-Agent Systems as\nNetworked Dynamical Systems",
            font_size=44, color=C["text"], line_spacing=1.1
        )
        subtitle = Text(
            "Workflow Evolution as Controller Synthesis",
            font_size=28, color=C["control"]
        )
        subtitle.next_to(title, DOWN, buff=0.6)

        deco = VGroup()
        for i, (x, col) in enumerate(zip([-3.5, 0, 3.5], [C["agent_a"], C["agent_b"], C["agent_c"]])):
            deco.add(make_contours(np.array([x, -2.4, 0]), col, n=4, r_max=0.9, seed=i))
        arrows = VGroup(
            Arrow(np.array([-2.5, -2.4, 0]), np.array([-1.0, -2.4, 0]), color=C["message"], stroke_width=2, buff=0),
            Arrow(np.array([1.0, -2.4, 0]), np.array([2.5, -2.4, 0]), color=C["message"], stroke_width=2, buff=0),
        )

        title.shift(UP * 1.2)
        subtitle.shift(UP * 1.2)

        self.play(Write(title), run_time=1.2)
        self.play(Write(subtitle), run_time=0.8)
        self.play(
            LaggedStart(*[Create(d) for d in deco], lag_ratio=0.2),
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3),
            run_time=1.5
        )
        self.wait(1.2)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_state_space_duality(self):
        """One system, two state spaces. The parametric view is a stated choice."""
        title = Text("1. One System, Two State Spaces", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.45)
        self.play(Write(title), run_time=0.8)

        # ---------- Left panel: full state (KV cache) ----------
        left_panel = RoundedRectangle(width=6.0, height=4.3, corner_radius=0.2,
                                      color=C["full"], fill_opacity=0.07, stroke_width=2)
        left_panel.move_to(LEFT * 3.4 + UP * 0.15)
        left_head = Text("Full state: the KV cache", font_size=22, color=C["full"])
        left_head.next_to(left_panel.get_top(), DOWN, buff=0.18)

        # growing token stack
        tokens = VGroup()
        for i in range(7):
            tk = Rectangle(width=0.5, height=0.32, color=C["full"],
                           fill_opacity=0.35, stroke_width=1)
            tokens.add(tk)
        tokens.arrange(RIGHT, buff=0.08)
        tokens.move_to(left_panel.get_center() + UP * 0.75)
        dots_more = MathTex(r"\cdots", font_size=26, color=C["full"])
        dots_more.next_to(tokens, RIGHT, buff=0.12)

        l1 = Text("context = initial condition", font_size=19, color=C["text"])
        l2 = MathTex(r"\text{dynamics fixed by } \theta \text{ — one landscape}",
                     font_size=24, color=C["text"])
        l3 = Text("exact, but dimension grows with t:\nno fixed manifold, no usable geometry",
                  font_size=17, color=C["grey"], line_spacing=0.9)
        lgroup = VGroup(l1, l2, l3).arrange(DOWN, buff=0.28)
        lgroup.move_to(left_panel.get_center() + DOWN * 0.65)

        self.play(Create(left_panel), Write(left_head), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in tokens],
                              lag_ratio=0.12), FadeIn(dots_more), run_time=1.1)
        self.play(LaggedStart(*[Write(m) for m in lgroup], lag_ratio=0.25), run_time=1.4)

        # ---------- Right panel: reduced state ----------
        right_panel = RoundedRectangle(width=6.0, height=4.3, corner_radius=0.2,
                                       color=C["agent_a"], fill_opacity=0.07, stroke_width=2)
        right_panel.move_to(RIGHT * 3.4 + UP * 0.15)
        right_head = Text("Reduced state: semantic state h", font_size=22, color=C["agent_a"])
        right_head.next_to(right_panel.get_top(), DOWN, buff=0.18)

        mini_manifold = Ellipse(width=2.4, height=1.3, color=C["agent_a"],
                                fill_opacity=0.15, stroke_width=2)
        mini_manifold.move_to(right_panel.get_center() + UP * 0.75)
        h_dot = Dot(mini_manifold.get_center() + RIGHT * 0.35, color=C["text"], radius=0.08)
        h_lab = MathTex("h_t", font_size=22, color=C["text"])
        h_lab.next_to(h_dot, UP, buff=0.08)

        ctx_bar = Rectangle(width=2.6, height=0.4, color=C["message"],
                            fill_opacity=0.4, stroke_width=1.5)
        ctx_bar.move_to(right_panel.get_center() + DOWN * 0.45)
        ctx_lab = Text("context c", font_size=16, color=C["message"])
        ctx_lab.move_to(ctx_bar.get_center())

        r1 = Text("attention re-reads c at every step", font_size=18, color=C["message"])
        r1.next_to(ctx_bar, DOWN, buff=0.22)
        r2 = MathTex(r"\Rightarrow\; c \text{ enters as a sustained parameter: } \Phi(h;\,c)",
                     font_size=23, color=C["control"])
        r2.next_to(r1, DOWN, buff=0.2)

        self.play(Create(right_panel), Write(right_head), run_time=0.9)
        self.play(Create(mini_manifold), FadeIn(h_dot), Write(h_lab),
                  FadeIn(ctx_bar), Write(ctx_lab), run_time=1.1)

        # attention pulses: context re-read at every generation step
        for _ in range(3):
            pulse = Dot(ctx_bar.get_top(), color=C["message"], radius=0.06)
            self.add(pulse)
            self.play(pulse.animate.move_to(h_dot.get_center()), run_time=0.4)
            self.remove(pulse)
        self.play(Write(r1), run_time=0.7)
        self.play(Write(r2), run_time=0.9)

        # ---------- Bottom: the duality, and our choice ----------
        duality = MathTex(
            r"\text{same system: non-autonomous} \;\leftrightarrow\; \text{autonomous embedding}",
            font_size=25, color=C["text"]
        )
        duality.to_edge(DOWN, buff=0.75)
        choice = Text(
            "We adopt the reduced view — it exposes attractors, coupling, and control structure",
            font_size=20, color=C["control"]
        )
        choice.next_to(duality, DOWN, buff=0.18)

        self.play(Write(duality), run_time=1)
        self.play(Write(choice), run_time=0.9)

        # small mechanistic footnote
        foot = Text("(support: attention ≈ descent on a context-built energy; ICL ≈ implicit ΔW)",
                    font_size=15, color=C["grey"])
        foot.next_to(choice, DOWN, buff=0.12)
        self.play(FadeIn(foot), run_time=0.6)

        self.wait(1.6)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_parametric_coupling(self):
        title = Text("2. Coupling is Parametric, not Additive", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        sub = Text("(in the reduced state space)", font_size=19, color=C["grey"])
        sub.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), FadeIn(sub), run_time=0.9)

        wrong = MathTex(
            r"\dot h_B = -\nabla \Phi_B(h_B) + K\,(h_A - h_B)",
            font_size=26, color=C["grey"]
        )
        wrong.shift(UP * 1.75)
        wrong_x = Text("✗ diffusive / force coupling", font_size=20, color=C["error"])
        wrong_x.next_to(wrong, RIGHT, buff=0.4)

        right = MathTex(
            r"\dot h_B = -\nabla \Phi_B\big(h_B\,;\, \pi_A(h_A^{\text{final}})\big)",
            font_size=28, color=C["text"]
        )
        right.next_to(wrong, DOWN, buff=0.35)
        right_c = Text("✓ message re-parameterizes B's field", font_size=20, color=C["good"])
        right_c.next_to(right, RIGHT, buff=0.4)

        self.play(Write(wrong), FadeIn(wrong_x), run_time=1)
        self.play(Write(right), FadeIn(right_c), run_time=1)

        # Visual: B's landscape deforms when the message arrives
        center_b1 = np.array([-2.2, -1.8, 0])
        center_b2 = np.array([-0.6, -1.5, 0])
        basin_b = make_contours(center_b1, C["agent_b"], n=5, r_max=1.25, seed=1)
        att_b = Dot(center_b1, color=C["agent_b"], radius=0.1)
        b_label = MathTex(r"\Phi_B(\cdot)", font_size=24, color=C["agent_b"])
        b_label.next_to(basin_b, LEFT, buff=0.2)

        self.play(
            LaggedStart(*[Create(c) for c in basin_b], lag_ratio=0.08),
            GrowFromCenter(att_b), Write(b_label),
            run_time=1.3
        )

        msg = VGroup(
            Rectangle(width=1.1, height=0.55, color=C["message"], fill_opacity=0.5, stroke_width=2),
            MathTex(r"m_A", font_size=20, color=WHITE)
        )
        msg[1].move_to(msg[0].get_center())
        msg.move_to(np.array([3.6, -0.6, 0]))
        msg_arrow = Arrow(msg.get_left(), center_b2 + RIGHT * 1.4, color=C["message"], stroke_width=2, buff=0.1)

        self.play(FadeIn(msg), run_time=0.5)
        self.play(GrowArrow(msg_arrow), run_time=0.7)

        basin_b2 = make_contours(center_b2, C["agent_b"], n=5, r_max=0.85, squash=0.55, seed=2)
        att_b2 = Dot(center_b2, color=C["agent_b"], radius=0.1)
        b_label2 = MathTex(r"\Phi_B(\cdot\,;m_A)", font_size=24, color=C["agent_b"])
        b_label2.next_to(basin_b2, RIGHT, buff=0.25)

        self.play(
            Transform(basin_b, basin_b2),
            Transform(att_b, att_b2),
            Transform(b_label, b_label2),
            run_time=2
        )

        note = Text("Attractors relocate, basins reshape — invisible in the KV-cache view",
                    font_size=21, color=C["control"])
        note.to_edge(DOWN, buff=0.5)
        self.play(Write(note), run_time=1)

        self.wait(1.3)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_slow_fast_drift(self):
        """Drift as slow parameter drift; derailment as saddle-node bifurcation."""
        title = Text("3. Drift is a Slow–Fast System", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.45)
        self.play(Write(title), run_time=0.8)

        eqs = VGroup(
            MathTex(r"\text{fast:}\;\; \dot h = -\nabla_h \Phi(h;\, a)", font_size=25, color=C["fast"]),
            MathTex(r"\text{slow:}\;\; \dot a = \epsilon \cdot (\text{context accumulation})", font_size=25, color=C["slow"]),
        ).arrange(RIGHT, buff=1.0)
        eqs.next_to(title, DOWN, buff=0.3)
        self.play(Write(eqs), run_time=1.2)

        # Potential axes
        axes = Axes(
            x_range=[-1.8, 1.8, 0.5], y_range=[-1.0, 1.5, 0.5],
            x_length=8.2, y_length=3.7,
            axis_config={"color": C["grey"], "stroke_width": 1.2,
                         "include_ticks": False},
        ).shift(DOWN * 0.85)
        y_lab = MathTex(r"\Phi(h;a)", font_size=22, color=C["text"])
        y_lab.next_to(axes, LEFT, buff=0.1).shift(UP * 1.0)

        a_vals = [0.0, 0.20, 0.35, 0.50]
        curves = [axes.plot(lambda x, a=a: potential(x, a), x_range=[-1.75, 1.75],
                            color=C["text"], stroke_width=3) for a in a_vals]

        self.play(Create(axes), Write(y_lab), Create(curves[0]), run_time=1.3)

        # ball in left (task) well
        x_l = left_min(0.0)
        ball = Dot(axes.c2p(x_l, potential(x_l, 0.0)) + UP * 0.09,
                   color=C["good"], radius=0.1)
        task_lab = Text("task attractor", font_size=17, color=C["good"])
        task_lab.next_to(axes.c2p(-1.0, potential(-1.0, 0.0)), DOWN, buff=0.25)

        x_r0 = right_min(0.0)
        ctx_lab = Text("context-consistency\nattractor", font_size=16, color=C["error"], line_spacing=0.85)
        ctx_lab.next_to(axes.c2p(x_r0, potential(x_r0, 0.0)), DOWN, buff=0.25)

        self.play(FadeIn(ball, scale=0.5), Write(task_lab), Write(ctx_lab), run_time=1)

        # context progress bar (the slow variable)
        bar_frame = Rectangle(width=3.2, height=0.32, color=C["slow"], stroke_width=2)
        bar_frame.to_corner(UR, buff=0.7).shift(DOWN * 0.9)
        bar_lab = Text("accumulated context a(t)", font_size=15, color=C["slow"])
        bar_lab.next_to(bar_frame, UP, buff=0.1)
        self.play(Create(bar_frame), Write(bar_lab), run_time=0.7)

        def bar_fill(frac):
            f = Rectangle(width=3.2 * frac, height=0.32, color=C["slow"],
                          fill_opacity=0.7, stroke_width=0)
            f.align_to(bar_frame, LEFT).align_to(bar_frame, DOWN)
            return f

        fill = bar_fill(0.05)
        self.add(fill)

        # Stage 1 -> 2 -> 3: gradual drift (well shallows, minimum shifts)
        drift_lab = Text("gradual degradation = slow drift of the attractor",
                         font_size=19, color=C["control"])
        drift_lab.to_edge(DOWN, buff=0.4)

        for i, (a, frac) in enumerate(zip(a_vals[1:3], [0.4, 0.7])):
            x_new = left_min(a)
            new_ball_pos = axes.c2p(x_new, potential(x_new, a)) + UP * 0.09
            anims = [
                Transform(curves[0], curves[i + 1]),
                ball.animate.move_to(new_ball_pos),
                Transform(fill, bar_fill(frac)),
            ]
            if i == 0:
                anims.append(Write(drift_lab))
            self.play(*anims, run_time=1.6)

        self.wait(0.4)

        # Stage 4: bifurcation — left well annihilated, ball rolls to right well
        bif_lab = Text("bifurcation: task attractor annihilated",
                       font_size=21, color=C["error"])
        bif_lab.to_edge(DOWN, buff=0.4)

        self.play(
            Transform(curves[0], curves[3]),
            Transform(fill, bar_fill(0.97)),
            FadeOut(drift_lab),
            run_time=1.2
        )
        self.play(Write(bif_lab), Flash(ball, color=C["error"], flash_radius=0.35), run_time=0.8)

        # roll along the a=0.5 curve to the right minimum
        a_f = 0.50
        x_start = left_min(0.35)  # ball's current x (old equilibrium, now gone)
        x_end = right_min(a_f)
        roll_pts = [axes.c2p(x, potential(x, a_f)) + UP * 0.09
                    for x in np.linspace(x_start, x_end, 40)]
        roll_path = VMobject()
        roll_path.set_points_smoothly(roll_pts)

        self.play(MoveAlongPath(ball, roll_path), run_time=1.8, rate_func=rush_into)
        self.play(ball.animate.set_color(C["error"]), run_time=0.4)

        punch = Text(
            "Sudden derailment is a bifurcation, not noise —\na discrete failure mode the static view cannot express",
            font_size=20, color=C["control"], line_spacing=1.0
        )
        punch.to_edge(DOWN, buff=0.35)
        self.play(FadeOut(bif_lab), Write(punch), run_time=1.2)

        self.wait(1.7)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_network_of_systems(self):
        title = Text("4. A Workflow is a Network of Coupled Systems", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        centers = [np.array([-4.2, -0.2, 0]), np.array([0, -0.2, 0]), np.array([4.2, -0.2, 0])]
        cols = [C["agent_a"], C["agent_b"], C["agent_c"]]
        names = ["A", "B", "C"]

        basins = VGroup(); atts = VGroup(); labels = VGroup()
        for ctr, col, nm, sd in zip(centers, cols, names, [0, 1, 2]):
            b = make_contours(ctr, col, n=4, r_max=1.15, seed=sd)
            basins.add(b)
            atts.add(Dot(ctr, color=col, radius=0.09))
            lab = Text(nm, font_size=24, color=col)
            lab.next_to(b, UP, buff=0.15)
            labels.add(lab)

        self.play(
            LaggedStart(*[Create(b) for b in basins], lag_ratio=0.2),
            LaggedStart(*[GrowFromCenter(a) for a in atts], lag_ratio=0.2),
            LaggedStart(*[Write(l) for l in labels], lag_ratio=0.2),
            run_time=2
        )

        rng = np.random.default_rng(5)
        def flow_into(ctr, start_offset):
            pts = [ctr + start_offset]
            cur = pts[0].copy()
            for i in range(9):
                d = ctr - cur
                d = d / (np.linalg.norm(d) + 1e-6)
                noise = rng.normal(0, 0.06, 3); noise[2] = 0
                cur = cur + 0.22 * d + noise
                pts.append(cur.copy())
            pts.append(ctr)
            v = VMobject(color=C["text"], stroke_width=2.5)
            v.set_points_smoothly(pts)
            return v

        flow_a = flow_into(centers[0], np.array([-1.0, 0.9, 0]))
        self.play(Create(flow_a), run_time=1.1)

        msg1 = Dot(centers[0] + RIGHT * 0.2, color=C["message"], radius=0.09)
        self.play(FadeIn(msg1), run_time=0.3)
        self.play(msg1.animate.move_to(centers[1] + LEFT * 1.0), run_time=0.8)
        jump1 = Text("discrete event", font_size=16, color=C["message"])
        jump1.next_to(midpoint(centers[0], centers[1]), UP, buff=0.6)
        self.play(Write(jump1), FadeOut(msg1), run_time=0.5)

        flow_b = flow_into(centers[1], np.array([-0.9, 0.8, 0]))
        self.play(Create(flow_b), run_time=1.1)

        msg2 = Dot(centers[1] + RIGHT * 0.2, color=C["message"], radius=0.09)
        self.play(FadeIn(msg2), run_time=0.3)
        self.play(msg2.animate.move_to(centers[2] + LEFT * 1.0), run_time=0.8)
        self.play(FadeOut(msg2), run_time=0.3)

        flow_c = flow_into(centers[2], np.array([-0.9, 0.7, 0]))
        self.play(Create(flow_c), run_time=1.1)

        hybrid = Text("Hybrid system: continuous flow within rollouts,\ndiscrete jumps at message events",
                      font_size=22, color=C["text"], line_spacing=1.0)
        hybrid.to_edge(DOWN, buff=0.9)
        skew = MathTex(
            r"\text{DAG topology} \;\Rightarrow\; \text{skew-product flow: base drives fiber}",
            font_size=24, color=C["control"]
        )
        skew.next_to(hybrid, DOWN, buff=0.2)

        self.play(Write(hybrid), run_time=1)
        self.play(Write(skew), run_time=0.9)

        self.wait(1.3)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_control_mapping(self):
        title = Text("5. The Control-Theoretic Dictionary", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        rows = [
            ("Worker agent", "Plant", C["agent_a"]),
            ("Planner", "Reference / feedforward", C["agent_b"]),
            ("Critic / verifier", "Sensor + feedback controller", C["critic"]),
            ("Orchestrator", "Supervisory controller", C["control"]),
            ("Pipeline topology", "Open-loop system", C["grey"]),
            ("Critique loop", "Closed-loop system", C["good"]),
        ]

        table = VGroup()
        for left_txt, right_txt, col in rows:
            left = Text(left_txt, font_size=24, color=col)
            arrow = MathTex(r"\longleftrightarrow", font_size=28, color=C["text"])
            right = Text(right_txt, font_size=24, color=C["text"])
            table.add(VGroup(left, arrow, right))

        table.arrange(DOWN, buff=0.42)
        for row in table:
            row[0].align_to(np.array([-5.8, 0, 0]), LEFT)
            row[1].move_to(np.array([-0.6, row[0].get_center()[1], 0]))
            row[2].align_to(np.array([0.6, 0, 0]), LEFT)
            row[2].set_y(row[0].get_center()[1])
        table.center().shift(DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in table], lag_ratio=0.2), run_time=2.6)

        note = Text("Evolution doesn't steer trajectories — it synthesizes the controller structure",
                    font_size=22, color=C["control"])
        note.to_edge(DOWN, buff=0.5)
        self.play(Write(note), run_time=1)

        self.wait(1.5)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_open_vs_closed_loop(self):
        title = Text("6. Why Critique Loops Beat Pipelines", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        open_label = Text("Open loop (pipeline): disturbance accumulates", font_size=22, color=C["error"])
        open_label.shift(UP * 2.2)
        self.play(Write(open_label), run_time=0.7)

        nodes_o = VGroup()
        for x, col, nm in zip([-4, -1.3, 1.3, 4],
                              [C["agent_a"], C["agent_b"], C["agent_c"], C["grey"]],
                              ["A", "B", "C", "out"]):
            n = VGroup(
                Circle(radius=0.42, color=col, fill_opacity=0.3, stroke_width=2),
                Text(nm, font_size=20, color=col)
            )
            n[1].move_to(n[0].get_center())
            n.move_to(np.array([x, 1.2, 0]))
            nodes_o.add(n)

        arrows_o = VGroup(*[
            Arrow(nodes_o[i][0].get_right(), nodes_o[i + 1][0].get_left(),
                  color=C["grey"], stroke_width=2, buff=0.08)
            for i in range(3)
        ])

        self.play(
            LaggedStart(*[FadeIn(n) for n in nodes_o], lag_ratio=0.15),
            LaggedStart(*[GrowArrow(a) for a in arrows_o], lag_ratio=0.15),
            run_time=1.4
        )

        err_sizes = [0.25, 0.55, 0.95, 1.4]
        err_bars = VGroup()
        for n, s in zip(nodes_o, err_sizes):
            bar = Rectangle(width=0.35, height=s, color=C["error"], fill_opacity=0.75, stroke_width=1)
            bar.next_to(n, DOWN, buff=0.12)
            err_bars.add(bar)
        err_label = MathTex(r"\|e\| \nearrow", font_size=24, color=C["error"])
        err_label.next_to(err_bars[-1], RIGHT, buff=0.3)

        self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in err_bars], lag_ratio=0.25), run_time=1.8)
        self.play(Write(err_label), run_time=0.5)

        closed_label = Text("Closed loop (worker + critic): disturbance rejected", font_size=22, color=C["good"])
        closed_label.shift(DOWN * 0.6)
        self.play(Write(closed_label), run_time=0.7)

        worker = VGroup(
            Circle(radius=0.5, color=C["agent_b"], fill_opacity=0.3, stroke_width=2),
            Text("worker", font_size=17, color=C["agent_b"])
        )
        worker[1].move_to(worker[0].get_center())
        worker.move_to(np.array([-1.6, -2.0, 0]))

        critic = VGroup(
            Circle(radius=0.5, color=C["critic"], fill_opacity=0.3, stroke_width=2),
            Text("critic", font_size=17, color=C["critic"])
        )
        critic[1].move_to(critic[0].get_center())
        critic.move_to(np.array([1.6, -2.0, 0]))

        fwd = Arrow(worker[0].get_right(), critic[0].get_left(), color=C["grey"], stroke_width=2, buff=0.08)
        fb = CurvedArrow(critic[0].get_bottom() + DOWN * 0.05, worker[0].get_bottom() + DOWN * 0.05,
                         angle=TAU / 4, color=C["critic"], stroke_width=2.5)
        fb_label = Text("error feedback", font_size=15, color=C["critic"])
        fb_label.next_to(fb, DOWN, buff=0.1)

        self.play(FadeIn(worker), FadeIn(critic), GrowArrow(fwd), run_time=1)
        self.play(Create(fb), Write(fb_label), run_time=0.9)

        it_sizes = [1.1, 0.6, 0.32, 0.15]
        it_bars = VGroup()
        for i, s in enumerate(it_sizes):
            bar = Rectangle(width=0.3, height=s, color=C["good"], fill_opacity=0.75, stroke_width=1)
            bar.move_to(np.array([3.6 + i * 0.55, -2.0 - (1.1 - s) / 2 + 0.4, 0]))
            it_bars.add(bar)
        it_label = MathTex(r"\|e\| \searrow", font_size=24, color=C["good"])
        it_label.next_to(it_bars, UP, buff=0.15)
        loop_lbl = Text("loop iterations", font_size=14, color=C["text"])
        loop_lbl.next_to(it_bars, DOWN, buff=0.12)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in it_bars], lag_ratio=0.3),
            Write(it_label), Write(loop_lbl),
            run_time=1.8
        )

        punch = Text("Drift is a disturbance in an open-loop system. Feedback rejects it.",
                     font_size=22, color=C["control"])
        punch.to_edge(DOWN, buff=0.35)
        self.play(Write(punch), run_time=1)

        self.wait(1.5)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_small_gain(self):
        title = Text("7. Stability: the Small-Gain Condition", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        cond = MathTex(
            r"\gamma_{\text{loop}} \;=\; \gamma_{\text{worker}} \cdot \gamma_{\text{critic}} \;<\; 1",
            font_size=34, color=C["control"]
        )
        cond.shift(UP * 1.6)
        self.play(Write(cond), run_time=1.2)

        axes = Axes(
            x_range=[0, 6, 1], y_range=[0, 3.4, 1],
            x_length=7.5, y_length=3.4,
            axis_config={"color": C["text"], "stroke_width": 1.5},
        ).shift(DOWN * 1.1)
        x_lab = Text("critique iterations", font_size=17, color=C["text"])
        x_lab.next_to(axes, DOWN, buff=0.15)
        y_lab = MathTex(r"\|e_n\|", font_size=22, color=C["text"])
        y_lab.next_to(axes, LEFT, buff=0.15).shift(UP * 1.2)

        conv = axes.plot(lambda x: 2.4 * (0.55 ** x), x_range=[0, 6], color=C["good"], stroke_width=3.5)
        div = axes.plot(lambda x: 0.35 * (1.45 ** x), x_range=[0, 5.4], color=C["error"], stroke_width=3.5)
        conv_l = MathTex(r"\gamma < 1", font_size=24, color=C["good"]).next_to(axes.c2p(4.6, 0.35), UP)
        div_l = MathTex(r"\gamma > 1", font_size=24, color=C["error"]).next_to(axes.c2p(4.3, 2.6), RIGHT)

        self.play(Create(axes), Write(x_lab), Write(y_lab), run_time=1.2)
        self.play(Create(conv), Write(conv_l), run_time=1.1)
        self.play(Create(div), Write(div_l), run_time=1.1)

        note = Text("Measurable: inject a controlled error, run one critique cycle, measure the residual",
                    font_size=20, color=C["text"])
        note.to_edge(DOWN, buff=0.35)
        self.play(Write(note), run_time=1)

        self.wait(1.4)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_observability(self):
        title = Text("8. Feedback Requires Observability", font_size=34, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        state_space = Circle(radius=1.6, color=C["agent_b"], fill_opacity=0.12, stroke_width=2)
        state_space.move_to(LEFT * 3.6 + DOWN * 0.4)
        ss_label = Text("worker state", font_size=18, color=C["agent_b"])
        ss_label.next_to(state_space, UP, buff=0.12)

        self.play(Create(state_space), Write(ss_label), run_time=1)

        channel = Rectangle(width=1.5, height=0.7, color=C["message"], fill_opacity=0.4, stroke_width=2)
        channel.move_to(np.array([0.2, -0.4, 0]))
        ch_label = MathTex(r"\pi \; (\text{message channel})", font_size=20, color=C["message"])
        ch_label.next_to(channel, UP, buff=0.12)

        critic = VGroup(
            Circle(radius=0.55, color=C["critic"], fill_opacity=0.3, stroke_width=2),
            Text("critic", font_size=17, color=C["critic"])
        )
        critic[1].move_to(critic[0].get_center())
        critic.move_to(np.array([3.8, -0.4, 0]))

        a1 = Arrow(state_space.get_right(), channel.get_left(), color=C["grey"], stroke_width=2, buff=0.1)
        a2 = Arrow(channel.get_right(), critic[0].get_left(), color=C["grey"], stroke_width=2, buff=0.1)

        self.play(FadeIn(channel), Write(ch_label), FadeIn(critic), GrowArrow(a1), GrowArrow(a2), run_time=1.4)

        e_obs = Arrow(state_space.get_center(), state_space.get_center() + RIGHT * 1.1,
                      color=C["good"], stroke_width=4, buff=0)
        e_obs_l = MathTex(r"e_{\text{obs}}", font_size=22, color=C["good"])
        e_obs_l.next_to(e_obs, DOWN, buff=0.08)

        self.play(GrowArrow(e_obs), Write(e_obs_l), run_time=0.8)
        pulse1 = Dot(state_space.get_center() + RIGHT * 1.1, color=C["good"], radius=0.08)
        self.play(pulse1.animate.move_to(critic[0].get_center()), run_time=1.0)
        seen = Text("✓ visible → correctable", font_size=18, color=C["good"])
        seen.next_to(critic, DOWN, buff=0.2)
        self.play(Write(seen), FadeOut(pulse1), run_time=0.6)

        e_unobs = Arrow(state_space.get_center(), state_space.get_center() + UP * 1.1,
                        color=C["error"], stroke_width=4, buff=0)
        e_unobs_l = MathTex(r"e \in \ker \pi", font_size=22, color=C["error"])
        e_unobs_l.next_to(e_unobs, LEFT, buff=0.08)

        self.play(GrowArrow(e_unobs), Write(e_unobs_l), run_time=0.8)
        pulse2 = Dot(state_space.get_center() + UP * 1.1, color=C["error"], radius=0.08)
        self.play(pulse2.animate.move_to(channel.get_center()), run_time=0.7)
        self.play(FadeOut(pulse2, scale=0.2), run_time=0.5)
        blocked = Text("✗ invisible → no controller can fix it", font_size=18, color=C["error"])
        blocked.next_to(channel, DOWN, buff=0.35)
        self.play(Write(blocked), run_time=0.7)

        unify = MathTex(
            r"\ker(\pi) \cap \mathcal{T}_{\text{crit}}(T) = \{0\}"
            r"\;\;\Longleftrightarrow\;\;"
            r"\text{task-critical errors are observable}",
            font_size=26, color=C["control"]
        )
        unify.to_edge(DOWN, buff=0.7)
        sub = Text("The information-bottleneck condition is an observability condition",
                   font_size=20, color=C["text"])
        sub.next_to(unify, DOWN, buff=0.2)

        self.play(Write(unify), run_time=1.3)
        self.play(Write(sub), run_time=0.8)

        self.wait(1.5)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_two_timescales(self):
        title = Text("9. Evolution = Slowest-Timescale Controller Synthesis", font_size=31, color=C["text"])
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        # Three nested timescales now: fast h, slow context drift, slowest evolution
        fast_box = RoundedRectangle(width=4.6, height=1.7, corner_radius=0.18,
                                    color=C["fast"], fill_opacity=0.14, stroke_width=2)
        fast_eq = MathTex(r"\dot h = -\nabla\Phi(h; a)", font_size=22, color=C["fast"])
        fast_t = Text("FAST: token dynamics", font_size=16, color=C["fast"])
        fast_t.next_to(fast_box.get_top(), DOWN, buff=0.1)
        fast_eq.move_to(fast_box.get_center() + DOWN * 0.15)

        mid_box = RoundedRectangle(width=7.2, height=3.1, corner_radius=0.22,
                                   color=C["control"], fill_opacity=0.08, stroke_width=2)
        mid_t = Text("SLOW: context drift within a rollout", font_size=16, color=C["control"])
        mid_t.next_to(mid_box.get_top(), DOWN, buff=0.1)
        mid_eq = MathTex(r"\dot a = \epsilon \,(\text{accumulation})", font_size=20, color=C["control"])
        mid_eq.next_to(mid_box.get_bottom(), UP, buff=0.15)

        slow_box = RoundedRectangle(width=10.2, height=4.7, corner_radius=0.25,
                                    color=C["slow"], fill_opacity=0.05, stroke_width=2.5)
        slow_t = Text("SLOWEST: workflow evolution across tasks", font_size=17, color=C["slow"])
        slow_t.next_to(slow_box.get_top(), DOWN, buff=0.1)
        slow_eq = MathTex(r"\mathcal{G}_{n+1} = \mathcal{M}(\mathcal{G}_n, \text{error signals}_n)",
                          font_size=21, color=C["slow"])
        slow_eq.next_to(slow_box.get_bottom(), UP, buff=0.15)

        nest = VGroup(slow_box, mid_box, fast_box)
        nest.arrange(ORIGIN)
        nest.shift(DOWN * 0.35)
        fast_t.next_to(fast_box.get_top(), DOWN, buff=0.08)
        fast_eq.move_to(fast_box.get_center() + DOWN * 0.1)
        mid_t.next_to(mid_box.get_top(), DOWN, buff=0.08)
        mid_eq.next_to(mid_box.get_bottom(), UP, buff=0.1)
        slow_t.next_to(slow_box.get_top(), DOWN, buff=0.08)
        slow_eq.next_to(slow_box.get_bottom(), UP, buff=0.1)

        self.play(Create(slow_box), Write(slow_t), run_time=0.9)
        self.play(Create(mid_box), Write(mid_t), run_time=0.9)
        self.play(Create(fast_box), Write(fast_t), Write(fast_eq), run_time=0.9)
        self.play(Write(mid_eq), Write(slow_eq), run_time=1)

        ilc = Text("Iterative learning control: repeat the task,\nupdate the controller structure from previous trials",
                   font_size=19, color=C["text"], line_spacing=1.0)
        ilc.to_edge(DOWN, buff=0.45)
        self.play(Write(ilc), run_time=1.1)

        self.wait(1.1)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.6)

        obj = MathTex(
            r"\mathcal{G}^* = \arg\max_{\mathcal{G}\in\mathcal{W}}\;"
            r"\Pr\big[\, h_0 \in \mathcal{B}_{\mathcal{G}}(\mathcal{A}_{\text{success}}) \,\big]",
            font_size=32, color=C["text"]
        )
        obj.shift(UP * 1.3)

        gloss = VGroup(
            MathTex(r"\mathcal{A}_{\text{success}}:\ \text{task-success attractor set}", font_size=24, color=C["good"]),
            MathTex(r"\mathcal{B}_{\mathcal{G}}:\ \text{its basin under closed-loop dynamics of } \mathcal{G}", font_size=24, color=C["control"]),
            Text("Evolution reshapes basins of attraction — and moves bifurcation points", font_size=23, color=C["slow"]),
        ).arrange(DOWN, buff=0.35)
        gloss.next_to(obj, DOWN, buff=0.7)

        self.play(Write(obj), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.2) for g in gloss], lag_ratio=0.3), run_time=1.8)

        self.wait(1.5)
        self.clear_all()

    # ------------------------------------------------------------------
    def scene_summary(self):
        title = Text("Summary: The Control-Theoretic Framework", font_size=36, color=C["text"])
        title.to_edge(UP, buff=0.55)

        points = VGroup(
            VGroup(
                Text("reduced state, context as parameter", font_size=21, color=C["agent_a"]),
                Text("A modeling choice, dual to the KV-cache view", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
            VGroup(
                MathTex(r"\Phi_B(\cdot\,; m_A)", font_size=24, color=C["message"]),
                Text("Messages re-parameterize downstream dynamics", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
            VGroup(
                Text("slow drift → bifurcation", font_size=21, color=C["error"]),
                Text("Gradual degradation vs. sudden derailment, one mechanism", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
            VGroup(
                MathTex(r"\gamma_{\text{loop}} < 1", font_size=24, color=C["good"]),
                Text("Critique loops = feedback; small gain ⇒ stability", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
            VGroup(
                MathTex(r"\ker(\pi)\cap\mathcal{T}_{\text{crit}} = \{0\}", font_size=24, color=C["control"]),
                Text("Bottleneck alignment = observability", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
            VGroup(
                MathTex(r"\max_{\mathcal{G}} \Pr[h_0 \in \mathcal{B}_{\mathcal{G}}(\mathcal{A}_{\text{succ}})]", font_size=24, color=C["slow"]),
                Text("Evolution: slowest-timescale controller synthesis", font_size=17, color=C["text"])
            ).arrange(RIGHT, buff=0.4),
        )
        points.arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        points.center().shift(DOWN * 0.25)

        self.play(Write(title), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(p, shift=RIGHT * 0.3) for p in points], lag_ratio=0.22), run_time=3)

        self.wait(2)


if __name__ == "__main__":
    import subprocess
    subprocess.run(["manim", "-pqm", __file__, "ControlTheoreticMASv2"])
