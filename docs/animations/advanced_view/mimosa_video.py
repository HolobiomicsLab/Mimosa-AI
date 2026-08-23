"""Captioned Manim video for the Mimosa framework document (v4 narrative).

Ten scenes, one per section of the document. Rendered individually and
concatenated by render.sh. No LaTeX: all math is Pango/Unicode text.
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


def formula(text, size=34, color=WHITE):
    """A math-looking Text mobject (Unicode, no LaTeX)."""
    return Text(text, font_size=size, color=color, slant=ITALIC)


def make_blob(center, radius, color, lobes=3, wobble=0.16, phase=0.0):
    """A closed wobbly blob representing one semantic stratum."""
    def point(t):
        r = radius * (1 + wobble * np.sin(lobes * t + phase))
        return np.array(center) + r * np.array([np.cos(t), np.sin(t), 0])
    blob = ParametricFunction(point, t_range=[0, TAU], color=color)
    blob.set_fill(color, opacity=0.15)
    blob.set_stroke(color, width=2.5)
    return blob


def wiggly_path(start, end, amplitude, waves, color):
    """A smooth wavy path from start to end (a 'trajectory')."""
    start, end = np.array(start), np.array(end)
    direction = end - start
    normal = np.array([-direction[1], direction[0], 0])
    normal = normal / max(np.linalg.norm(normal), 1e-9)

    def point(t):
        return start + t * direction + amplitude * np.sin(waves * PI * t) * normal
    return ParametricFunction(point, t_range=[0, 1], color=color, stroke_width=3)


class MimosaScene(Scene):
    """Base scene: crossfading bottom captions and a section header."""

    def setup(self):
        self._caption = None

    def cap(self, text, wait=3.0):
        """Show a caption at the bottom, crossfading from the previous one."""
        new = Text(text, font_size=CAPTION_SIZE, color=CAPTION_COLOR)
        if new.width > MAX_CAPTION_WIDTH:
            new.scale_to_fit_width(MAX_CAPTION_WIDTH)
        new.to_edge(DOWN, buff=0.32)
        anims = [FadeIn(new, shift=UP * 0.2)]
        if self._caption is not None:
            anims.append(FadeOut(self._caption, shift=UP * 0.2))
        self.play(*anims, run_time=0.6)
        self._caption = new
        self.wait(wait * 1.35 + 0.9)

    def clear_cap(self):
        if self._caption is not None:
            self.play(FadeOut(self._caption), run_time=0.4)
            self._caption = None

    def header(self, text):
        h = Text(text, font_size=30, weight=BOLD, color=HEADER_COLOR)
        h.to_edge(UP, buff=0.3)
        self.play(FadeIn(h, shift=DOWN * 0.2), run_time=0.6)
        return h


# ------------------------------------------------------------------ scene 01
class S01Title(MimosaScene):
    def construct(self):
        self.show_title_card()
        self.show_collapse_teaser()

    def show_title_card(self):
        title = Text("Why Language-Model Agents Fail", font_size=44, weight=BOLD)
        sub = Text("when decomposition helps, and what evolution actually learns",
                   font_size=27, color=GREY_A)
        brand = Text("The Mimosa framework · video overview", font_size=21, color=GREY_B)
        card = VGroup(title, sub, brand).arrange(DOWN, buff=0.45)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.0)
        self.play(FadeIn(sub), FadeIn(brand), run_time=0.8)
        self.wait(2.4)
        self.play(FadeOut(card), run_time=0.6)

    def show_collapse_teaser(self):
        good = wiggly_path([-6, 0.8, 0], [1.2, 0.5, 0], 0.25, 3, BLUE_C)
        spiral = self.collapse_spiral(np.array([3.3, -0.4, 0]))
        bridge = wiggly_path([1.2, 0.5, 0], spiral.get_start(), 0.2, 2, RED_C)
        dot = Dot(color=BLUE_C, radius=0.09).move_to(good.get_start())
        self.add(TracedPath(dot.get_center, stroke_color=BLUE_C, stroke_width=4))
        self.add(dot)
        self.cap("An agent starts a long task well…", wait=0.2)
        self.play(MoveAlongPath(dot, good, rate_func=linear), run_time=3.0)
        dot.set_color(RED_C)
        self.add(TracedPath(dot.get_center, stroke_color=RED_C, stroke_width=4))
        self.cap("…one early mistake — then fluent, confident elaboration of that mistake…", wait=0.2)
        self.play(MoveAlongPath(dot, bridge, rate_func=linear), run_time=1.4)
        self.play(MoveAlongPath(dot, spiral, rate_func=linear), run_time=3.2)
        self.play(Flash(dot, color=RED_C, flash_radius=0.5), run_time=0.8)
        self.cap("…until nothing is salvageable. Terminal reasoning collapse — this video explains it.", wait=3.0)

    @staticmethod
    def collapse_spiral(center):
        def point(t):
            r = 1.1 * np.exp(-0.55 * t)
            return center + r * np.array([np.cos(2.4 * t + PI), np.sin(2.4 * t + PI), 0])
        return ParametricFunction(point, t_range=[0, 4.5], color=RED_C)


# ------------------------------------------------------------------ scene 02
class S02FolkTheory(MimosaScene):
    def construct(self):
        self.header("1 · A folk theory that would settle everything")
        eq = formula("P(correct) = (1 − e)^n", size=40).shift(UP * 2.2)
        self.play(Write(eq), run_time=1.2)
        self.cap("Suppose every generated token independently derails the task with probability e.", wait=2.8)
        self.show_decay_plot()
        self.cap("Survival decays exponentially in length — LeCun's argument that autoregressive LLMs are doomed.", wait=3.0)
        self.show_broken_assumptions()

    def show_decay_plot(self):
        axes = Axes(x_range=[0, 500, 100], y_range=[0, 1, 0.25],
                    x_length=6.2, y_length=3.0,
                    axis_config={"include_tip": False, "font_size": 18})
        axes.shift(DOWN * 0.6 + LEFT * 2.6)
        curve = axes.plot(lambda n: 0.99 ** n, x_range=[0, 500], color=BAD)
        labels = VGroup(
            Text("tokens n", font_size=20, color=GREY_A).next_to(axes.x_axis, DOWN, buff=0.15),
            Text("e = 1%", font_size=22, color=BAD).next_to(axes, RIGHT, buff=0.3),
        )
        self.play(Create(axes), run_time=0.8)
        self.play(Create(curve), FadeIn(labels), run_time=1.6)

    def show_broken_assumptions(self):
        chips = VGroup(
            self.chip("errors are UNIFORM over positions"),
            self.chip("errors are INDEPENDENT across positions"),
        ).arrange(DOWN, buff=0.35).shift(RIGHT * 3.4 + DOWN * 0.4)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.3) for c in chips], lag_ratio=0.3), run_time=1.2)
        self.cap("The argument rests on two hidden assumptions.", wait=2.0)
        crosses = VGroup(*[Cross(c, stroke_width=5).scale(0.9) for c in chips])
        self.play(LaggedStart(*[Create(x) for x in crosses], lag_ratio=0.3), run_time=1.2)
        self.cap("Transformers violate both — and in opposite directions: errors are far more concentrated, and far more correlated.", wait=3.6)

    @staticmethod
    def chip(text):
        label = Text(text, font_size=22)
        box = SurroundingRectangle(label, color=GREY_B, buff=0.18, corner_radius=0.1)
        return VGroup(label, box)


# ------------------------------------------------------------------ scene 03
class S03TwoRate(MimosaScene):
    KEY_SLOTS = (5, 14, 22, 31)
    N_TOKENS = 36

    def construct(self):
        self.header("2 · The two-rate model — errors are sparse")
        self.show_token_row()
        self.show_two_rate_formula()
        self.show_regimes_plot()
        self.cap("Length is not the enemy. The decision budget is.", wait=3.2)

    def show_token_row(self):
        squares = VGroup(*[
            Square(0.27, fill_opacity=0.9,
                   fill_color=KEY if i in self.KEY_SLOTS else GREY_D,
                   stroke_width=1, stroke_color=GREY_B)
            for i in range(self.N_TOKENS)
        ]).arrange(RIGHT, buff=0.07).shift(UP * 2.1)
        self.play(LaggedStart(*[FadeIn(s, scale=0.6) for s in squares], lag_ratio=0.02), run_time=1.6)
        self.cap("In natural text, only ~5–10% of tokens genuinely depend on long-range context — the key decisions.", wait=3.0)
        keys = VGroup(*[squares[i] for i in self.KEY_SLOTS])
        self.play(Indicate(keys, color=KEY, scale_factor=1.35), run_time=1.0)
        self.cap("Perplexity on just these tokens predicts long-context performance at ρ ≈ −0.96; ordinary perplexity predicts nothing (Fang et al., 2024).", wait=3.6)

    def show_two_rate_formula(self):
        eq = formula("P ≈ (1 − e_key)^k · (1 − e_non)^(n−k)", size=34)
        cond = formula("k ≪ n,   e_non → 0", size=26, color=GREY_A)
        group = VGroup(eq, cond).arrange(DOWN, buff=0.25).shift(UP * 0.8)
        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(cond), run_time=0.6)
        self.cap("Two rates, not one: rare hard decisions, and filler the model gets right for free.", wait=2.8)
        self.play(group.animate.scale(0.75).to_edge(RIGHT, buff=0.5).shift(DOWN * 1.2), run_time=0.8)

    def show_regimes_plot(self):
        axes = Axes(x_range=[0, 600, 150], y_range=[0, 1, 0.25],
                    x_length=6.4, y_length=3.1,
                    axis_config={"include_tip": False, "font_size": 18})
        axes.shift(DOWN * 1.15 + LEFT * 2.9)
        folk = axes.plot(lambda n: 0.992 ** n, x_range=[0, 600], color=BAD)
        power = axes.plot(lambda n: (1 + n / 25) ** -0.5, x_range=[0, 600], color=YELLOW_C)
        plateau = axes.plot(lambda n: 0.78 + 0.16 * np.exp(-n / 90), x_range=[0, 600], color=GOOD)
        tags = VGroup(
            Text("uniform (folk)", font_size=18, color=BAD),
            Text("k ~ log n", font_size=18, color=YELLOW_C),
            Text("k bounded → plateau", font_size=18, color=GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(axes, UP, buff=0.1).shift(RIGHT * 2.2)
        self.play(Create(axes), run_time=0.7)
        self.play(Create(folk), Create(power), Create(plateau), FadeIn(tags), run_time=2.2)
        self.cap("If the number of genuine decisions is bounded, reliability plateaus — long trajectories are not intrinsically doomed.", wait=3.4)


# ------------------------------------------------------------------ scene 04
class S04ThreeSpaces(MimosaScene):
    def construct(self):
        self.header("3 · Three spaces — tokens, trajectories, solutions")
        prompt, solutions, paths = self.show_spaces()
        self.show_projection(paths)
        self.show_reachable_futures(paths[0], solutions)

    def show_spaces(self):
        prompt = LabeledDot(Text("x0", font_size=18), radius=0.22, color=WHITE)
        prompt.move_to([-5.6, 0, 0])
        solutions = VGroup(*[
            LabeledDot(Text(f"x{i + 1}", font_size=18), radius=0.26,
                       color=[TEAL_D, BLUE_D, PURPLE_D][i])
            for i in range(3)
        ])
        for i, s in enumerate(solutions):
            s.move_to([4.8, 1.7 - 1.7 * i, 0])
        paths = VGroup(
            wiggly_path(prompt.get_center(), solutions[1].get_center(), 0.55, 4, TEAL_C),
            wiggly_path(prompt.get_center(), solutions[1].get_center(), -0.85, 3, BLUE_C),
            wiggly_path(prompt.get_center(), solutions[1].get_center(), 1.15, 2, PURPLE_C),
        )
        space_tag = Text("solution space X", font_size=20, color=GREY_A)
        space_tag.next_to(solutions, UP, buff=0.3)
        self.play(FadeIn(prompt), FadeIn(solutions), FadeIn(space_tag), run_time=0.9)
        self.cap("A prompt, and the space of solutions the task actually cares about.", wait=1.8)
        self.play(LaggedStart(*[Create(p) for p in paths], lag_ratio=0.35), run_time=2.6)
        self.cap("Trajectories are token sequences — thousands of ways to word the same answer.", wait=2.4)
        return prompt, solutions, paths

    def show_projection(self, paths):
        phi = formula("φ : T → X", size=30, color=YELLOW_C).shift(UP * 2.5 + RIGHT * 0.5)
        self.play(Write(phi), run_time=0.8)
        self.play(*[p.animate.set_stroke(opacity=0.35) for p in paths[1:]], run_time=0.6)
        self.cap("The projection φ collapses trajectories to solutions: many wordings, one answer. Reward lives on X, never on tokens.", wait=3.2)

    def show_reachable_futures(self, path, solutions):
        dot = Dot(color=WHITE, radius=0.09).move_to(path.point_from_proportion(0.25))
        fans = self.fan(dot, solutions, [0.8, 0.9, 0.7])
        self.play(FadeIn(dot), Create(fans), run_time=1.0)
        self.cap("p_t — the reachable-futures distribution: where this could still end up, given what is committed so far.", wait=3.0)
        narrow = self.fan(Dot().move_to(path.point_from_proportion(0.7)), solutions, [0.12, 1.0, 0.12])
        self.play(dot.animate.move_to(path.point_from_proportion(0.7)),
                  Transform(fans, narrow), run_time=1.8)
        self.cap("Every committed token reshapes p_t. Drift = how far context has pushed it away from good solutions — measured by restart-and-resample.", wait=3.6)

    @staticmethod
    def fan(dot, solutions, weights):
        return VGroup(*[
            DashedLine(dot.get_center(), s.get_center(), dash_length=0.12,
                       stroke_opacity=w, stroke_width=3.5, color=GREY_A)
            for s, w in zip(solutions, weights)
        ])


# ------------------------------------------------------------------ scene 05
class S05LockIn(MimosaScene):
    def construct(self):
        self.header("4 · Sparse injection, continuous lock-in")
        self.show_fork()
        self.show_coherence_ratchet()
        self.show_escape_decay()

    def show_fork(self):
        stem = Line([-6, 0.5, 0], [-1.8, 0.5, 0], color=GREY_A, stroke_width=3)
        fork = Dot([-1.8, 0.5, 0], color=KEY, radius=0.12)
        target = Star(n=5, outer_radius=0.3, color=GOOD, fill_opacity=1).move_to([5.9, 1.9, 0])
        good = DashedLine(fork.get_center(), target.get_center(), color=GOOD, dash_length=0.15)
        bad = wiggly_path(fork.get_center(), [4.2, -0.3, 0], 0.3, 2, BAD)
        walker = Dot(color=BLUE_C, radius=0.09).move_to(stem.get_start())
        self.add(walker)
        self.play(Create(stem), FadeIn(fork), FadeIn(target), Create(good), run_time=1.2)
        self.cap("Errors enter at key decision points — forks where the sampled token selects among futures.", wait=0.4)
        self.play(MoveAlongPath(walker, stem, rate_func=linear), run_time=1.6)
        self.play(Flash(fork, color=KEY), run_time=0.6)
        walker.set_color(BAD)
        self.play(Create(bad), MoveAlongPath(walker, bad, rate_func=linear), run_time=2.2)
        self.cap("Mechanism A — injection: rare, stochastic, at identifiable junctions.", wait=2.4)
        self.walker = walker

    def show_coherence_ratchet(self):
        eq = formula("objective(t) = R(φ(τ)) + λ(t) · Coh(τ, c_t)", size=28)
        eq.move_to([0, 2.55, 0])
        pressure = ValueTracker(0.15)
        bar = always_redraw(lambda: Rectangle(
            width=0.45, height=max(pressure.get_value(), 0.01) * 2.6,
            fill_color=BAD, fill_opacity=0.85, stroke_width=0,
        ).move_to([6.2, -1.6, 0], aligned_edge=DOWN))
        bar_label = Text("λ(t)", font_size=22, color=BAD).move_to([6.2, -2.0, 0])
        self.play(Write(eq), FadeIn(bar_label), run_time=1.0)
        self.add(bar)
        self.cap("Mechanism B — lock-in: generation conditions on its own past output.", wait=0.4)
        self.play(pressure.animate.set_value(1.0), run_time=2.6)
        self.cap("The objective drifts from solving the task to staying consistent with what was already said. Same mistake, step 50 vs step 5000: different fates.", wait=3.6)

    def show_escape_decay(self):
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 1, 0.5],
                    x_length=3.6, y_length=2.0,
                    axis_config={"include_tip": False, "font_size": 16})
        axes.to_edge(LEFT, buff=0.5).shift(DOWN * 1.7)
        curve = axes.plot(lambda t: 0.65 ** t, x_range=[0, 10], color=BAD)
        tag = formula("P(escape) ~ ρ^t", size=24, color=BAD).next_to(axes, UP, buff=0.15)
        self.play(Create(axes), Create(curve), FadeIn(tag), run_time=1.6)
        self.cap("Proposition 1: after capture, escape probability decays geometrically — recovery needs an external kick.", wait=3.0)
        self.cap("One context is both memory and conditioning: every stored error becomes a self-reinforcing prior. Self-critique inherits the corruption it should catch.", wait=4.0)


# ------------------------------------------------------------------ scene 06
class S06Geometry(MimosaScene):
    def construct(self):
        self.header("5 · The geometry beneath")
        blobs = self.show_archipelago()
        self.show_contraction(blobs)
        self.show_straightening()
        self.show_itinerancy()

    def show_archipelago(self):
        blobs = VGroup(
            make_blob([-3.7, 0.9, 0], 1.35, BLUE_D, lobes=3, phase=0.4),
            make_blob([0.2, -0.9, 0], 1.1, TEAL_D, lobes=4, phase=1.6),
            make_blob([3.8, 1.0, 0], 1.25, PURPLE_D, lobes=3, phase=2.7),
        )
        names = VGroup(
            Text("code", font_size=22, color=BLUE_B).move_to(blobs[0]),
            Text("prose", font_size=22, color=TEAL_B).move_to(blobs[1]),
            Text("math", font_size=22, color=PURPLE_B).move_to(blobs[2]),
        )
        self.play(LaggedStart(*[Create(b) for b in blobs], lag_ratio=0.25),
                  FadeIn(names), run_time=2.2)
        self.cap("Representation space is an archipelago: low-dimensional semantic strata, not a uniform fog (Li & Sarwate, 2025).", wait=3.2)
        self.blob_names = names
        return blobs

    def show_contraction(self, blobs):
        keep = blobs[0]
        self.play(FadeOut(blobs[1:]), FadeOut(self.blob_names),
                  keep.animate.scale(1.6).move_to([-3.2, 0.4, 0]), run_time=1.0)
        attractor = Dot(keep.get_center(), color=YELLOW, radius=0.11)
        starts = [keep.get_center() + 1.5 * np.array([np.cos(a), np.sin(a), 0])
                  for a in np.linspace(0.3, TAU, 5, endpoint=False)]
        probes = VGroup(*[Dot(p, color=WHITE, radius=0.06) for p in starts])
        self.play(FadeIn(attractor), FadeIn(probes), run_time=0.7)
        self.cap("Within a stratum, layers act as contractions: paraphrases collapse toward a concept attractor (Chytas & Singh).", wait=0.4)
        self.play(*[p.animate.move_to(attractor.get_center() + 0.12 * (p.get_center() - attractor.get_center()))
                    for p in probes], run_time=2.2)
        self.wait(1.6)
        self.contract_group = VGroup(keep, attractor, probes)

    def show_straightening(self):
        curved = wiggly_path([1.2, 1.6, 0], [6.2, 1.6, 0], 0.6, 4, GREY_A)
        straight = Line([1.2, 1.6, 0], [6.2, 1.6, 0], color=WHITE, stroke_width=3.5)
        tag = Text("layer depth →", font_size=20, color=GREY_B).next_to(straight, UP, buff=0.2)
        self.play(Create(curved), FadeIn(tag), run_time=1.0)
        self.cap("And trained models straighten trajectories — prediction by linear extrapolation (Hosseini & Fedorenko, NeurIPS 2023).", wait=0.4)
        self.play(Transform(curved, straight), run_time=1.8)
        self.wait(1.4)
        self.straighten_group = VGroup(curved, tag)

    def show_itinerancy(self):
        self.play(FadeOut(self.contract_group), FadeOut(self.straighten_group), run_time=0.6)
        basin = Ellipse(width=4.6, height=2.0, color=TEAL_C).shift(DOWN * 0.4)
        inward = VGroup(
            Arrow(basin.get_top() + UP * 0.7, basin.get_top(), buff=0.05, color=GOOD),
            Arrow(basin.get_bottom() + DOWN * 0.7, basin.get_bottom(), buff=0.05, color=GOOD),
        )
        outward = Arrow(basin.get_center(), basin.get_right() + RIGHT * 1.3,
                        buff=0.1, color=BAD)
        labels = VGroup(
            Text("transverse: contract", font_size=20, color=GOOD).next_to(inward[0], UP, buff=0.5).shift(UP*0.1),
            Text("tangential: expand", font_size=20, color=BAD).next_to(outward, DOWN, buff=0.15),
        )
        self.play(Create(basin), Create(inward), Create(outward), FadeIn(labels), run_time=1.8)
        self.cap("Transverse contraction, tangential expansion: the system itinerates — rides a basin, then exits into a neighbor.", wait=3.2)
        self.cap("A hallucination is an unplanned basin exit, followed by confident contraction into the wrong basin.", wait=3.0)
        self.cap("Key claim: key decisions ARE basin routing events — visible as entropy spikes in ordinary API logprobs.", wait=3.4)


# ------------------------------------------------------------------ scene 07
class S07Boundary(MimosaScene):
    def construct(self):
        self.header("6 · What a boundary really is")
        self.show_handoff()
        self.show_benefits()
        self.show_costs()
        self.cap("Every framework reduces to this one operator — and most analyses count only the green column.", wait=3.2)

    def show_handoff(self):
        self.box_a = self.agent_box("agent A", [-4.6, 1.7, 0], BLUE_D)
        self.box_b = self.agent_box("agent B", [4.6, 1.7, 0], TEAL_D)
        self.lattice = self.icl_lattice(self.box_a.get_center())
        doc = self.document_glyph([0, 1.7, 0])
        arrows = VGroup(
            Arrow(self.box_a.get_right(), doc.get_left(), buff=0.15, color=GREY_A),
            Arrow(doc.get_right(), self.box_b.get_left(), buff=0.15, color=GREY_A),
        )
        op = formula("B = ι ∘ φ   (lossy jump operator)", size=24, color=YELLOW_C)
        op.next_to(doc, DOWN, buff=0.3)
        self.play(FadeIn(self.box_a), FadeIn(self.box_b), FadeIn(self.lattice), run_time=0.9)
        self.cap("Agent A works, then writes something down. Agent B starts fresh from that write-up.", wait=0.4)
        self.play(FadeIn(doc), Create(arrows), Write(op), run_time=1.6)
        self.wait(1.8)

    def show_benefits(self):
        items = self.column([
            "B1  lock-in reset:  λ → 0",
            "B2  a scheduled external kick",
            "B3  fresh context ⇒ decorrelated verification",
        ], GOOD, [-3.6, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in items], lag_ratio=0.3), run_time=1.5)
        self.cap("The buys: coherence pressure resets, frozen decisions re-open, and a fresh verifier has independent errors.", wait=3.2)
        self.cap("Independence, not intelligence, makes verification work — so same-model critics are correlated witnesses.", wait=3.2)

    def show_costs(self):
        items = self.column([
            "C1  in-context structure destroyed",
            "C2  beliefs freeze into premises",
            "C3  a new routing decision",
        ], BAD, [3.6, -0.4, 0])
        self.play(FadeIn(items[0], shift=LEFT * 0.3),
                  FadeOut(self.lattice, scale=0.3), run_time=1.2)
        self.cap("The costs: A's in-context-learned structure cannot cross the token bottleneck…", wait=2.2)
        self.show_cliff_plot()
        self.play(FadeIn(items[1], shift=LEFT * 0.3), run_time=0.8)
        self.cap("…A's revisable beliefs arrive in B as hard premises — errors don't reset, they harden…", wait=2.8)
        self.play(FadeIn(items[2], shift=LEFT * 0.3),
                  Flash(self.box_b.get_left(), color=KEY), run_time=1.0)
        self.cap("…and B's re-contraction into a basin is itself a new key decision that can misroute.", wait=2.8)
        self.play(FadeOut(self.cliff), run_time=0.4)

    def show_cliff_plot(self):
        axes = Axes(x_range=[0, 10, 5], y_range=[0, 1, 0.5],
                    x_length=2.9, y_length=1.6,
                    axis_config={"include_tip": False, "font_size": 14})
        axes.move_to([0.2, -2.5, 0])
        step = axes.plot(lambda x: 0.06 if x < 5.5 else 0.92,
                         x_range=[0, 10], color=KEY, discontinuities=[5.5], use_smoothing=False)
        tag = Text("handoff mass → cliff at x_c (Park et al., ICLR 2025)",
                   font_size=16, color=GREY_A).next_to(axes, UP, buff=0.08)
        self.cliff = VGroup(axes, step, tag)
        self.play(Create(axes), Create(step), FadeIn(tag), run_time=1.4)
        self.cap("…and re-establishing it is a phase transition: below a critical handoff mass, B stays on pretrained priors entirely.", wait=3.2)

    @staticmethod
    def agent_box(name, position, color):
        box = RoundedRectangle(corner_radius=0.15, width=2.6, height=1.7,
                               color=color, stroke_width=3)
        label = Text(name, font_size=22, color=color).next_to(box, UP, buff=0.12)
        return VGroup(box, label).move_to(position)

    @staticmethod
    def icl_lattice(center):
        pts = [center + np.array([0.5 * i - 0.5, 0.4 * j - 0.4, 0])
               for i in range(3) for j in range(3)]
        dots = VGroup(*[Dot(p, radius=0.045, color=YELLOW_B) for p in pts])
        edges = VGroup(*[Line(pts[i], pts[i + 3], stroke_width=1.5, color=YELLOW_E)
                         for i in range(6)])
        return VGroup(edges, dots)

    @staticmethod
    def document_glyph(position):
        page = RoundedRectangle(corner_radius=0.08, width=1.0, height=1.3,
                                color=GREY_A, stroke_width=2.5)
        lines = VGroup(*[Line(LEFT * 0.32, RIGHT * 0.32, stroke_width=2, color=GREY_B)
                         .shift(UP * (0.3 - 0.3 * i)) for i in range(3)])
        return VGroup(page, lines).move_to(position)

    @staticmethod
    def column(texts, color, position):
        items = VGroup(*[Text(t, font_size=21, color=color) for t in texts])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(position)
        return items


# ------------------------------------------------------------------ scene 08
class S08Placement(MimosaScene):
    RIDGE_X = 0.0
    MIDBASIN_X = -2.7

    def construct(self):
        self.header("7 · The placement principle")
        self.show_landscape()
        self.show_cuts()
        self.show_tagline()

    def landscape_point(self, x):
        u = x / 1.9
        return np.array([x, 0.13 * (u * u - 4) ** 2 - 1.9, 0])

    def show_landscape(self):
        curve = ParametricFunction(lambda x: self.landscape_point(x),
                                   t_range=[-5.4, 5.4], color=GREY_A, stroke_width=3.5)
        ball = Dot(color=BLUE_C, radius=0.1).move_to(self.landscape_point(-3.8))
        ride = ParametricFunction(lambda x: self.landscape_point(x),
                                  t_range=[-3.8, 3.8], color=BLUE_C)
        spike = Text("entropy spike", font_size=19, color=KEY)
        spike.next_to(self.landscape_point(self.RIDGE_X), UP, buff=0.5)
        self.play(Create(curve), FadeIn(ball), run_time=1.4)
        self.cap("Run the task single-agent, instrumented: the trajectory rides one basin, then crosses a ridge.", wait=0.4)
        self.play(MoveAlongPath(ball, ride, rate_func=linear), run_time=3.0)
        self.play(FadeIn(spike, shift=DOWN * 0.2),
                  Flash(self.landscape_point(self.RIDGE_X), color=KEY), run_time=1.0)
        self.cap("The crossing is loud in the logprobs — an entropy spike marks the basin transition.", wait=2.6)
        self.spike = spike

    def show_cuts(self):
        good_cut = DashedLine(self.landscape_point(self.RIDGE_X) + UP * 0.4,
                              self.landscape_point(self.RIDGE_X) + UP * 1.6,
                              color=GOOD, stroke_width=4)
        good_tag = Text("ΔJ > 0  ✓", font_size=24, color=GOOD).next_to(good_cut, RIGHT, buff=0.2)
        bad_cut = DashedLine(self.landscape_point(self.MIDBASIN_X) + UP * 0.4,
                             self.landscape_point(self.MIDBASIN_X) + UP * 1.6,
                             color=BAD, stroke_width=4)
        bad_tag = Text("ΔJ < 0  ✗", font_size=24, color=BAD).next_to(bad_cut, LEFT, buff=0.2)
        eq = formula("ΔJ(b) = (Reset + Unlock + Verif) − (Residue + Freeze + Route)", size=25)
        eq.to_edge(UP, buff=1.05)
        self.play(Write(eq), run_time=1.2)
        self.play(Create(good_cut), FadeIn(good_tag), run_time=0.9)
        self.cap("A boundary at the transition adds no new decision — the routing was happening anyway — and destroys the least structure.", wait=3.4)
        self.play(Create(bad_cut), FadeIn(bad_tag), run_time=0.9)
        self.cap("A mid-basin boundary destroys live structure, freezes mid-thought beliefs, and adds a decision the task never required.", wait=3.4)
        self.cut_marks = VGroup(good_cut, good_tag, bad_cut, bad_tag)

    def show_tagline(self):
        self.play(FadeOut(self.cut_marks), FadeOut(self.spike), run_time=0.6)
        tag = Text("Cut where the dynamics already want to jump.",
                   font_size=32, weight=BOLD, color=YELLOW_B).move_to([0, 1.55, 0])
        backdrop = SurroundingRectangle(tag, color=YELLOW_D, buff=0.25, corner_radius=0.12)
        self.play(FadeIn(tag, scale=1.1), Create(backdrop), run_time=1.0)
        self.cap("RL found the same principle from the other side: options begin and end at bottleneck states (Sutton–Precup–Singh; McGovern–Barto).", wait=3.6)


# ------------------------------------------------------------------ scene 09
class S09Evolution(MimosaScene):
    TRUE_CUTS = (-1.3, 2.1)
    AXIS_HALF = 4.4

    def construct(self):
        self.header("8 · What evolution actually learns")
        self.show_population()
        self.run_generations()
        self.show_lessons()

    def show_population(self):
        self.rows = VGroup(*[self.workflow_row(y, cuts) for y, cuts in [
            (1.9, [-3.4]), (0.9, [0.4, 3.3]), (-0.1, [-2.6, 1.0]), (-1.1, [-1.0, 2.6]),
        ]])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in self.rows],
                              lag_ratio=0.2), run_time=1.8)
        self.cap("A population of candidate workflows: same task, different boundary placements.", wait=2.4)
        self.cap("The topology→performance map is black-box, noisy, expensive — exactly where population search earns its keep.", wait=3.0)

    def workflow_row(self, y, cut_positions):
        task_line = Line([-self.AXIS_HALF, y, 0], [self.AXIS_HALF, y, 0],
                         color=GREY_D, stroke_width=5)
        cuts = VGroup(*[Line([x, y - 0.22, 0], [x, y + 0.22, 0],
                             color=BAD, stroke_width=5) for x in cut_positions])
        return VGroup(task_line, cuts)

    def run_generations(self):
        losers, winners = [self.rows[0], self.rows[1]], [self.rows[2], self.rows[3]]
        self.play(*[r.animate.set_opacity(0.18) for r in losers], run_time=0.8)
        self.cap("Selection dims the misplaced cuts…", wait=1.6)
        moves = []
        for row in winners:
            for cut in row[1]:
                target = min(self.TRUE_CUTS, key=lambda t: abs(t - cut.get_center()[0]))
                moves.append(cut.animate.shift(RIGHT * (target - cut.get_center()[0])))
        ticks = VGroup(*[Line([x, -1.8, 0], [x, 2.3, 0], color=YELLOW_C,
                              stroke_width=2).set_opacity(0.55) for x in self.TRUE_CUTS])
        self.play(*moves, run_time=2.4)
        self.play(Create(ticks), run_time=1.0)
        self.cap("…and surviving boundaries drift onto the model's real basin transitions: system identification, from rollouts alone.", wait=3.4)

    def show_lessons(self):
        self.cap("Start minimal and complexify (NEAT): add a boundary only when fitness pays for its cost.", wait=3.0)
        self.cap("The evolved archive is a learned prior over decompositions — a map of the model's geometry, bought with compute.", wait=3.4)


# ------------------------------------------------------------------ scene 10
class S10Predictions(MimosaScene):
    LINES = (
        "P1  Crossover — basin sharpness predicts the single- vs multi-agent sign",
        "P2  Placement — evolved cuts land on single-agent entropy spikes",
        "P3  Coupling — boundary discontinuity D anti-correlates with fitness",
        "P4  Threshold — handoffs fail as a cliff, not a slope",
        "P5  Verifiers — cross-model checks beat same-model at equal capability",
    )

    def construct(self):
        h = self.header("9 · How this framework can fail")
        items = VGroup(*[Text(t, font_size=24) for t in self.LINES])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.42).shift(UP * 0.5)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in items],
                              lag_ratio=0.35), run_time=3.0)
        self.cap("Every construct has a logprob-level estimator; P1–P4 are answerable from already-logged traces.", wait=3.2)
        self.cap("If P2 fails, the geometry demotes to heuristic — and the two-rate accounting stands on its own.", wait=3.2)
        self.show_end_card(h, items)

    def show_end_card(self, header, items):
        self.clear_cap()
        self.play(FadeOut(items), FadeOut(header), run_time=0.8)
        tag = Text("A workflow is a controlled-itinerancy schedule.",
                   font_size=36, weight=BOLD, color=YELLOW_B)
        brand = Text("Mimosa · framework draft v4", font_size=24, color=GREY_A)
        card = VGroup(tag, brand).arrange(DOWN, buff=0.6)
        self.play(FadeIn(tag, scale=1.08), run_time=1.2)
        self.play(FadeIn(brand), run_time=0.8)
        self.wait(3.0)
        self.play(FadeOut(card), run_time=1.0)


SCENES = [S01Title, S02FolkTheory, S03TwoRate, S04ThreeSpaces, S05LockIn,
          S06Geometry, S07Boundary, S08Placement, S09Evolution, S10Predictions]

if __name__ == "__main__":
    print(f"{len(SCENES)} scenes defined:", ", ".join(s.__name__ for s in SCENES))
