"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { arrow, arrowDown, drawMatrix, label, line, outline, peakOf, shade, DIM, HI } from "@/lib/cnn/draw";
import matrix from "@/lib/cnn/matrix";
import type { Model, Trace } from "@/lib/cnn";

const W = 680;
const H = 380;

type Scene = {
  name: string;
  caption: string;
  ms: number;
  draw: (ctx: CanvasRenderingContext2D, u: number) => void;
};

// Pick the filter to star in the walkthrough. Scoring on activation alone
// tends to surface a filter with almost no negative response, which makes the
// ReLU step look like it does nothing -- so weight both halves.
function pickFilter(pre: Array<matrix>): number {
  let best = 0;
  let bestScore = -Infinity;
  for (let f = 0; f < pre.length; f++) {
    const m = pre[f];
    let pos = 0;
    let neg = 0;
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        const v = m.values[i][j];
        if (v > 0) pos += v;
        else neg -= v;
      }
    }
    const score = pos * Math.sqrt(neg);
    if (score > bestScore) {
      bestScore = score;
      best = f;
    }
  }
  return best;
}

function vecPeak(v: Array<number>): number {
  let p = 0;
  for (const x of v) p = Math.max(p, Math.abs(x));
  return p || 1;
}

function drawVector(
  ctx: CanvasRenderingContext2D,
  v: Array<number>,
  x: number,
  y: number,
  w: number,
  h: number,
  upto: number,
  peak: number,
) {
  const cw = w / v.length;
  ctx.fillStyle = "#000";
  ctx.fillRect(x, y, w, h);
  const n = Math.min(upto, v.length);
  for (let i = 0; i < n; i++) {
    if (v[i] === 0) continue;
    // Gamma, like a log axis: ReLU leaves these vectors very sparse, and on a
    // linear ramp one dominant unit renders every other one as black.
    const g = Math.sign(v[i]) * Math.pow(Math.abs(v[i]) / peak, 0.55) * peak;
    ctx.fillStyle = shade(g, peak);
    ctx.fillRect(x + i * cw, y, Math.max(cw, 0.7), h);
  }
}

function buildScenes(
  input: matrix,
  model: Model,
  trace: Trace,
  probs: Array<number>,
): Array<Scene> {
  const L1 = trace[0];
  const L2 = trace[1];
  const f0 = pickFilter(L1.pre);
  const kernel = model.layers[0].filters[f0][0];
  const pre = L1.pre[f0];
  const act = L1.maps[f0];
  const pooled = L1.pooled[f0];

  // The dense half isn't in the trace -- recomputing it costs ~100k MACs once
  // per prediction, which is cheaper than threading two more arrays through.
  const flat: Array<number> = [];
  for (const m of L2.pooled) flat.push(...m.flatten());
  const X = new matrix(flat.map((v) => [v]));
  const B1 = new matrix(model.b1.map((v) => [v]));
  const h = model.W1.mult(X).add(B1).flatten().map((v) => Math.max(0, v));

  const hLive = h.filter((v) => v > 0).length;
  const prePeak = peakOf(pre);
  const flatPeak = vecPeak(flat);
  const hPeak = vecPeak(h);
  const winner = probs.indexOf(Math.max(...probs));

  return [
    {
      name: "input",
      caption:
        "Your drawing is cropped to its ink, scaled to fit a 20×20 box, and centred by its centre of mass into a 28×28 grid — exactly how MNIST was built. Each cell is one number from 0 (black) to 1 (white).",
      ms: 2400,
      draw: (ctx, u) => {
        const cell = 11;
        const x = (W - 28 * cell) / 2;
        const y = 46;
        drawMatrix(ctx, input, x, y, cell, { upto: Math.ceil(u * 784), peak: 1 });
        outline(ctx, x, y, 28 * cell, 28 * cell, "#333");
        label(ctx, "input · 28×28 · one channel", x, y - 18);
      },
    },
    {
      name: "convolve",
      caption:
        "A 3×3 kernel slides over the image one pixel at a time. At each stop it multiplies the nine pixels underneath by its nine weights and adds them up — that single number becomes one pixel of the feature map. 26×26 stops, because a 3×3 window only fits 26 times across 28.",
      ms: 6400,
      draw: (ctx, u) => {
        const ic = 9;
        const ix = 28;
        const iy = 62;
        const ox = 424;
        const oy = 62;

        // Linger on the first row, then accelerate -- the mechanism is only
        // legible while it's slow, but 676 stops at that pace is unwatchable.
        const p =
          u < 0.28
            ? Math.floor((u / 0.28) * 26)
            : Math.floor(26 + ((u - 0.28) / 0.72) * (676 - 26));
        const oi = Math.min(25, Math.floor(p / 26));
        const oj = Math.min(25, p % 26);

        drawMatrix(ctx, input, ix, iy, ic, { peak: 1 });
        outline(ctx, ix, iy, 28 * ic, 28 * ic, "#333");
        label(ctx, "input · 28×28", ix, iy - 18);

        drawMatrix(ctx, pre, ox, oy, ic, { upto: p, peak: prePeak });
        outline(ctx, ox, oy, 26 * ic, 26 * ic, "#333");
        label(ctx, `feature map · 26×26 · filter ${f0}`, ox, oy - 18);

        // kernel, with its actual learned weights
        const kc = 34;
        const kx = 296;
        const ky = 138;
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            const v = kernel.values[i][j];
            ctx.fillStyle = shade(v, peakOf(kernel));
            ctx.fillRect(kx + j * kc, ky + i * kc, kc, kc);
            label(
              ctx,
              v.toFixed(2),
              kx + j * kc + kc / 2,
              ky + i * kc + kc / 2 - 4,
              Math.abs(v) / peakOf(kernel) > 0.55 ? "#000" : "#bbb",
              10,
              "center",
            );
          }
        }
        outline(ctx, kx, ky, 3 * kc, 3 * kc, "#555");
        label(ctx, "kernel · 3×3", kx, ky - 18);
        label(
          ctx,
          `Σ = ${pre.values[oi][oj].toFixed(3)}`,
          kx + 1.5 * kc,
          ky + 3 * kc + 12,
          HI,
          13,
          "center",
        );

        // receptive field on the input, and where it lands on the output
        const wx = ix + oj * ic;
        const wy = iy + oi * ic;
        outline(ctx, wx, wy, 3 * ic, 3 * ic, HI, 1.5);
        const cx = ox + oj * ic;
        const cy = oy + oi * ic;
        outline(ctx, cx, cy, ic, ic, HI, 1.5);
        line(ctx, wx + 3 * ic, wy + 1.5 * ic, kx, ky + 1.5 * kc, DIM, 0.5);
        line(ctx, kx + 3 * kc, ky + 1.5 * kc, cx, cy + ic / 2, DIM, 0.5);
      },
    },
    {
      name: "relu",
      caption:
        "Those sums can be negative (blue). ReLU clamps every negative to zero, leaving only the evidence that the filter's pattern was present — never evidence against it. This is the entire nonlinearity.",
      ms: 3000,
      draw: (ctx, u) => {
        const cell = 11;
        const x = (W - 26 * cell) / 2;
        const y = 50;
        const w = 26 * cell;

        drawMatrix(ctx, pre, x, y, cell, { peak: prePeak });
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w * u, 26 * cell);
        ctx.clip();
        drawMatrix(ctx, pre, x, y, cell, { peak: prePeak, relu: true });
        ctx.restore();

        outline(ctx, x, y, w, 26 * cell, "#333");
        if (u > 0.01 && u < 0.99) {
          line(ctx, x + w * u, y, x + w * u, y + 26 * cell, HI, 0.9);
        }
        const below = y + 26 * cell + 12;
        label(ctx, "after ReLU", x + w * 0.25, below, HI, 12, "center");
        label(ctx, "before · negatives in blue", x + w * 0.75, below, "#6f97cc", 12, "center");
      },
    },
    {
      name: "pool",
      caption:
        "Max pooling walks 2×2 blocks and keeps only the largest value in each. The map halves to 13×13 — cheaper, and a feature that shifts by a pixel still lands in the same bucket.",
      ms: 3600,
      draw: (ctx, u) => {
        const ac = 9;
        const ax = 52;
        const ay = 66;
        const pc = 18;
        const px = 394;
        const py = 66;
        const p = Math.floor(u * 169);
        const pi = Math.min(12, Math.floor(p / 13));
        const pj = Math.min(12, p % 13);

        drawMatrix(ctx, act, ax, ay, ac, { peak: prePeak });
        outline(ctx, ax, ay, 26 * ac, 26 * ac, "#333");
        label(ctx, "after ReLU · 26×26", ax, ay - 18);

        drawMatrix(ctx, pooled, px, py, pc, { upto: p, peak: prePeak });
        outline(ctx, px, py, 13 * pc, 13 * pc, "#333");
        label(ctx, "pooled · 13×13", px, py - 18);

        outline(ctx, ax + pj * 2 * ac, ay + pi * 2 * ac, 2 * ac, 2 * ac, HI, 1.5);
        outline(ctx, px + pj * pc, py + pi * pc, pc, pc, HI, 1.5);
        arrow(ctx, ax + 26 * ac + 22, ay + 13 * ac, 64);
        label(ctx, "max", ax + 26 * ac + 54, ay + 13 * ac - 20, DIM, 11, "center");
      },
    },
    {
      name: "filters",
      caption:
        "That was one filter. Layer 1 runs 32 of them in parallel, each with different weights, so each responds to a different pattern — edges at various angles, ends of strokes, curves. Layer 2's 64 filters then convolve across all 32 of those channels at once and sum the results, so they compose simple parts into larger shapes.",
      ms: 4600,
      draw: (ctx, u) => {
        if (u < 0.5) {
          const a = u / 0.5;
          const c = 4;
          const s = 13 * c;
          const gap = 8;
          const gx = (W - (8 * s + 7 * gap)) / 2;
          const gy = 76;
          const n = Math.ceil(a * 32);
          for (let i = 0; i < n; i++) {
            drawMatrix(ctx, L1.pooled[i], gx + (i % 8) * (s + gap), gy + Math.floor(i / 8) * (s + gap), c, {
              alpha: Math.min(1, (a * 32 - i) * 1.5),
            });
          }
          label(ctx, "conv 1 · 32 filters · 13×13 each", gx, gy - 18);
        } else {
          const a = (u - 0.5) / 0.5;
          const c = 7;
          const s = 5 * c;
          const gap = 5;
          const gx = (W - (16 * s + 15 * gap)) / 2;
          const gy = 110;
          const n = Math.ceil(a * 64);
          for (let i = 0; i < n; i++) {
            drawMatrix(ctx, L2.pooled[i], gx + (i % 16) * (s + gap), gy + Math.floor(i / 16) * (s + gap), c, {
              alpha: Math.min(1, (a * 64 - i) * 1.5),
            });
          }
          label(ctx, "conv 2 · 64 filters · 5×5 each", gx, gy - 18);
        }
      },
    },
    {
      name: "flatten",
      caption:
        "The 64 maps of 5×5 are read out in order into a single list of 1600 numbers. Nothing is computed here — the grid structure is simply dropped, and from this point on the network sees a flat vector.",
      ms: 3400,
      draw: (ctx, u) => {
        const c = 6;
        const s = 5 * c;
        const gap = 4;
        const gx = (W - (16 * s + 15 * gap)) / 2;
        const gy = 52;
        const shown = Math.ceil(u * 64);
        for (let i = 0; i < 64; i++) {
          drawMatrix(ctx, L2.pooled[i], gx + (i % 16) * (s + gap), gy + Math.floor(i / 16) * (s + gap), c, {
            alpha: i < shown ? 0.28 : 1,
          });
        }
        label(ctx, "conv 2 output · 64 × 5×5", gx, gy - 18);

        const sx = 40;
        const sy = 246;
        const sw = W - 80;
        arrowDown(ctx, W / 2, 200, 26);
        drawVector(ctx, flat, sx, sy, sw, 34, Math.ceil(u * 1600), flatPeak);
        outline(ctx, sx, sy, sw, 34, "#333");
        label(ctx, `flattened · ${Math.ceil(u * 1600)} / 1600`, sx, sy - 18);
      },
    },
    {
      name: "dense",
      caption:
        "Two fully connected layers finish the job. Every one of the 1600 values feeds all 64 hidden units (that's the 64×1600 weight matrix), ReLU again, then those 64 feed 10 output scores — one per digit. Notice how few hidden units survive the ReLU: this network leans on a handful of them.",
      ms: 4200,
      draw: (ctx, u) => {
        const sx = 40;
        const sw = W - 80;
        drawVector(ctx, flat, sx, 60, sw, 22, 1600, flatPeak);
        outline(ctx, sx, 60, sw, 22, "#333");
        label(ctx, "1600", sx, 42);

        const hx = 100;
        const hw = W - 200;
        const hShown = u < 0.55 ? Math.ceil((u / 0.55) * 64) : 64;
        drawVector(ctx, h, hx, 180, hw, 22, hShown, hPeak);
        outline(ctx, hx, 180, hw, 22, "#333");
        label(ctx, `64 hidden · ReLU · ${hLive} still nonzero`, hx, 162);

        // sampled connections, not all 102,400 of them
        const fan = u < 0.55 ? u / 0.55 : 1;
        for (let i = 0; i < 64; i += 2) {
          if (i / 64 > fan) break;
          line(ctx, sx + ((i + 0.5) / 64) * sw, 82, hx + ((i + 0.5) / 64) * hw, 180, DIM, 0.16);
        }

        if (u > 0.55) {
          const b = (u - 0.55) / 0.45;
          const ox = 190;
          const ow = W - 380;
          const oShown = Math.ceil(b * 10);
          drawVector(ctx, probs, ox, 300, ow, 24, oShown, 1);
          outline(ctx, ox, 300, ow, 24, "#333");
          label(ctx, "10 scores", ox, 282);
          for (let i = 0; i < 10; i++) {
            if (i / 10 > b) break;
            line(ctx, hx + ((i * 6 + 3) / 64) * hw, 202, ox + ((i + 0.5) / 10) * ow, 300, DIM, 0.22);
          }
        }
      },
    },
    {
      name: "softmax",
      caption:
        "Softmax exponentiates the 10 scores and divides by their total, turning them into probabilities that sum to 1. The largest one is the answer.",
      ms: 3600,
      draw: (ctx, u) => {
        const bx = 210;
        const bw = 380;
        const rowH = 27;
        const top = 50;
        for (let i = 0; i < 10; i++) {
          const y = top + i * rowH;
          const on = i === winner;
          label(ctx, String(i), bx - 26, y + 3, on ? HI : DIM, 14);
          ctx.fillStyle = "#1c1c1c";
          ctx.fillRect(bx, y, bw, 16);
          ctx.fillStyle = on ? HI : "#4a4a4a";
          ctx.fillRect(bx, y, bw * probs[i] * Math.min(1, u * 1.6), 16);
          label(
            ctx,
            `${(probs[i] * 100).toFixed(2)}%`,
            bx + bw + 10,
            y + 3,
            on ? HI : DIM,
            11,
          );
        }
        if (u > 0.28) {
          ctx.save();
          ctx.globalAlpha = Math.min(1, (u - 0.28) * 4);
          label(ctx, String(winner), 100, 120, HI, 96, "center");
          label(ctx, "prediction", 100, 232, DIM, 11, "center");
          ctx.restore();
        }
      },
    },
  ];
}

export default function CnnAnimation({
  input,
  model,
  trace,
  probs,
}: {
  input: matrix;
  model: Model;
  trace: Trace;
  probs: Array<number>;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const clock = useRef(0);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [done, setDone] = useState(false);

  const scenes = useMemo(
    () => buildScenes(input, model, trace, probs),
    [input, model, trace, probs],
  );
  const total = useMemo(() => scenes.reduce((s, x) => s + x.ms, 0), [scenes]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    el.width = W * dpr;
    el.height = H * dpr;
    const ctx = el.getContext("2d");
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, []);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let last = performance.now();

    const frame = (now: number) => {
      const dt = now - last;
      last = now;
      if (playing) {
        clock.current += dt;
        if (clock.current >= total) {
          clock.current = total;
          setPlaying(false);
          setDone(true);
        }
      }

      let t = clock.current;
      let i = 0;
      while (i < scenes.length - 1 && t >= scenes[i].ms) {
        t -= scenes[i].ms;
        i++;
      }
      const u = Math.max(0, Math.min(1, t / scenes[i].ms));

      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, W, H);
      label(ctx, `${i + 1}/${scenes.length}  ${scenes[i].name}`, 16, 14, DIM);
      scenes[i].draw(ctx, u);

      if (bar.current) {
        bar.current.style.width = `${(clock.current / total) * 100}%`;
      }
      setIdx((prev) => (prev === i ? prev : i));

      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, scenes, total]);

  function jump(i: number) {
    let t = 0;
    for (let k = 0; k < i; k++) t += scenes[k].ms;
    clock.current = t;
    setIdx(i);
    setDone(false);
    setPlaying(true);
  }

  function toggle() {
    if (done) {
      clock.current = 0;
      setDone(false);
    }
    setPlaying((p) => !p);
  }

  return (
    <div className="flex w-full flex-col gap-3">
      <canvas
        ref={canvas}
        style={{ aspectRatio: `${W} / ${H}` }}
        className="w-full rounded-lg border bg-[#0a0a0a]"
      />

      <div className="bg-muted h-0.5 w-full overflow-hidden rounded-full">
        <div ref={bar} className="bg-foreground h-full w-0" />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" onClick={toggle}>
          {playing ? "Pause" : done ? "Replay" : "Play"}
        </Button>
        {scenes.map((s, i) => (
          <Button
            key={s.name}
            size="sm"
            variant={i === idx ? "secondary" : "ghost"}
            onClick={() => jump(i)}
          >
            {s.name}
          </Button>
        ))}
      </div>

      <p className="text-muted-foreground max-w-2xl text-sm leading-relaxed">
        {scenes[idx].caption}
      </p>
    </div>
  );
}
