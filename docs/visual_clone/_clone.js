
/* Mimosa-AI visual clone - shared behaviour. Offline, no dependencies. */
(function () {
  /* default stamp; a page overrides it with <body data-built="..." data-built-date="..."> (set from _manifest.json) */
  var BUILD = { commit: "ca72107+wt", date: "2026-09-28" };

  /* ---------- top bar ---------- */
  function topbar() {
    if (document.getElementById("topbar")) return;
    var b = document.body;
    var crumb = b.getAttribute("data-crumb") || "Overview";
    var bar = document.createElement("div");
    bar.id = "topbar";
    var parts = crumb.split(">");
    var html = '<div class="tb"><a class="menu" href="index.html">Menu</a>'
      + '<span class="crumb">' + parts.join(" \u203a ") + "</span>"
      + '<span class="built">built <b>' + (b.getAttribute("data-built") || BUILD.commit) + "</b> \u00b7 "
      + (b.getAttribute("data-built-date") || BUILD.date) + "</span></div>";
    bar.innerHTML = html;
    document.body.insertBefore(bar, document.body.firstChild);
  }

  /* ---------- legend (auto-filled when <div id="legend" data-auto>) ---------- */
  var LAYERS = [
    ["L0 planner", "--l0"], ["L1 tools/MCP", "--l1"], ["L2 evolution", "--l2"],
    ["L3 runner", "--l3"], ["L4 verifier", "--l4"], ["entry/config", "--ice"],
    ["utils", "--steel"], ["external", "--ext"]
  ];
  function legend() {
    var el = document.getElementById("legend");
    if (!el || el.getAttribute("data-auto") === null) return;
    var bits = LAYERS.map(function (l) {
      return '<span class="k" style="color:var(' + l[1] + ')"><span class="sw"></span>' + l[0] + "</span>";
    });
    bits.push('<span class="k">solid = call</span>');
    bits.push('<span class="k">dashed = reads/writes</span>');
    bits.push('<span class="k">dotted = import</span>');
    bits.push('<span class="k"><span class="shp"></span> file</span>');
    bits.push('<span class="k"><span class="shp r"></span> class</span>');
    bits.push('<span class="k"><span class="shp p"></span> function</span>');
    bits.push('<span class="k" style="color:var(--ok)">pass / ok</span>');
    bits.push('<span class="k" style="color:var(--warn)">warning / stale</span>');
    bits.push('<span class="k" style="color:var(--err)">fail</span>');
    el.innerHTML = bits.join("");
  }

  /* ---------- controls ---------- */
  function controls() {
    var c = document.createElement("div");
    c.id = "controls";
    c.innerHTML = '<button data-act="expand">expand all</button>'
      + '<button data-act="collapse">collapse all</button>'
      + '<button data-act="print">print / pdf</button>';
    document.body.appendChild(c);
    c.addEventListener("click", function (e) {
      var a = e.target.getAttribute("data-act");
      if (!a) return;
      if (a === "print") { window.print(); return; }
      var open = a === "expand";
      document.querySelectorAll("details").forEach(function (d) { d.open = open; });
    });
  }

  /* ---------- copy buttons ---------- */
  function copyButtons() {
    function mk(target, text) {
      var btn = document.createElement("button");
      btn.className = "copybtn";
      btn.textContent = "copy";
      btn.addEventListener("click", function (ev) {
        ev.preventDefault(); ev.stopPropagation();
        copy(text, function (okFlag) {
          btn.textContent = okFlag ? "copied" : "failed";
          btn.className = "copybtn" + (okFlag ? " done" : "");
          setTimeout(function () { btn.textContent = "copy"; btn.className = "copybtn"; }, 1200);
        });
      });
      if (target.nextSibling) target.parentNode.insertBefore(btn, target.nextSibling);
      else target.parentNode.appendChild(btn);
    }
    document.querySelectorAll("code.path, code.uuid, [data-copy]").forEach(function (el) {
      var text = el.getAttribute("data-copy") || el.textContent.trim();
      mk(el, text);
    });
  }
  function copy(text, cb) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { cb(true); }, function () { cb(fallback(text)); });
    } else { cb(fallback(text)); }
  }
  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  /* ---------- diagram hover: highlight connected edges ---------- */
  function diagrams() {
    document.querySelectorAll("svg.diagram").forEach(function (svg) {
      var boxes = svg.querySelectorAll("a.box, g.box");
      boxes.forEach(function (box) {
        box.addEventListener("mouseenter", function () {
          var id = box.getAttribute("id") || (box.parentNode && box.parentNode.getAttribute("id"));
          svg.classList.add("hl");
          svg.querySelectorAll("path.edge").forEach(function (p) {
            var connected = (p.getAttribute("data-from") === id || p.getAttribute("data-to") === id);
            p.classList.toggle("dim", !connected);
          });
          var shape = box.querySelector("rect, circle");
          if (shape) shape.style.filter = "url(#glow)";
        });
        box.addEventListener("mouseleave", function () {
          svg.classList.remove("hl");
          svg.querySelectorAll("path.edge").forEach(function (p) { p.classList.remove("dim"); });
          var shape = box.querySelector("rect, circle");
          if (shape) shape.style.filter = "";
        });
      });
    });
  }

  /* ---------- freshness badge helper (index.html) ---------- */
  window.CLONE_BUILD = BUILD;


  /* ---------- menu search (index.html) ---------- */
  function search() {
    var box = document.getElementById("search");
    if (!box) return;
    box.addEventListener("input", function () {
      var q = box.value.trim().toLowerCase();
      var hits = 0;
      document.querySelectorAll("[data-name]").forEach(function (el) {
        var match = !q || el.getAttribute("data-name").toLowerCase().indexOf(q) !== -1;
        el.style.display = match ? "" : "none";
        if (match) hits++;
        if (match && q) {
          var d = el.closest("details");
          while (d) { d.open = true; d = d.parentElement ? d.parentElement.closest("details") : null; }
        }
      });
      var out = document.getElementById("searchcount");
      if (out) out.textContent = q ? hits + " pages match" : "";
      document.querySelectorAll("section.treesection").forEach(function (s) {
        var any = Array.prototype.some.call(s.querySelectorAll("[data-name]"), function (e) { return e.style.display !== "none"; });
        s.style.display = any ? "" : "none";
      });
    });
  }

  /* ---------- print: show every collapsed block ---------- */
  function printExpand() {
    var opened = [];
    document.querySelectorAll("details").forEach(function (d) { if (!d.open) { d.open = true; opened.push(d); } });
    window.addEventListener("afterprint", function () {
      opened.forEach(function (d) { d.open = false; });
    }, { once: true });
  }
  function init() { topbar(); legend(); controls(); copyButtons(); diagrams(); search(); }
  window.addEventListener("beforeprint", printExpand);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
