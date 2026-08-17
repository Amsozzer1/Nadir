"use client";

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { infer, loadWeights, predict, toInput, type Model } from "@/lib/cnn";

const SIZE = 280;
const STROKE = 22;

export default function DigitCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [model, setModel] = useState<Model | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probs, setProbs] = useState<Array<number> | null>(null);
  const [ms, setMs] = useState(0);

  useEffect(() => {
    loadWeights().then(setModel, (e) => setError(e?.message ?? String(e)));
  }, []);

  function stroke(e: React.PointerEvent<HTMLCanvasElement>, begin: boolean) {
    const c = ref.current?.getContext("2d");
    if (!c) return;
    const { offsetX: x, offsetY: y } = e.nativeEvent;
    if (begin) {
      c.lineWidth = STROKE;
      c.lineCap = "round";
      c.lineJoin = "round";
      c.strokeStyle = "#fff";
      c.beginPath();
      c.moveTo(x, y);
    }
    c.lineTo(x, y);
    c.stroke();
  }

  function down(e: React.PointerEvent<HTMLCanvasElement>) {
    drawing.current = true;
    ref.current?.setPointerCapture(e.pointerId);
    stroke(e, true);
  }

  function move(e: React.PointerEvent<HTMLCanvasElement>) {
    if (drawing.current) stroke(e, false);
  }

  function clear() {
    ref.current?.getContext("2d")?.clearRect(0, 0, SIZE, SIZE);
    setProbs(null);
  }

  function run() {
    if (!ref.current || !model) return;
    const input = toInput(ref.current);
    if (!input) {
      setProbs(null);
      return;
    }
    const t = performance.now();
    const p = infer(model, input);
    setMs(performance.now() - t);
    setProbs(p);
  }

  return (
    <div className="flex flex-wrap items-start gap-6">
      <div className="flex flex-col gap-2">
        <canvas
          ref={ref}
          width={SIZE}
          height={SIZE}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={() => (drawing.current = false)}
          onPointerCancel={() => (drawing.current = false)}
          className="touch-none rounded-lg border bg-black"
        />
        <div className="flex gap-2">
          <Button onClick={run} disabled={!model}>
            {model ? "Predict" : "Loading weights…"}
          </Button>
          <Button variant="outline" onClick={clear}>
            Clear
          </Button>
        </div>
      </div>

      <div className="min-w-48 font-mono text-sm">
        {error && <p className="text-destructive">{error}</p>}
        {!error && !probs && (
          <p className="text-muted-foreground">Draw a digit, then Predict.</p>
        )}
        {probs && (
          <>
            <p className="mb-2 text-2xl">
              {predict(probs)}{" "}
              <span className="text-muted-foreground text-xs">
                {(Math.max(...probs) * 100).toFixed(1)}% · {ms.toFixed(0)}ms
              </span>
            </p>
            {probs.map((p, i) => (
              <div
                key={i}
                className={
                  i === predict(probs)
                    ? "flex justify-between gap-4"
                    : "text-muted-foreground flex justify-between gap-4"
                }
              >
                <span>{i}</span>
                <span>{(p * 100).toFixed(2)}%</span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
