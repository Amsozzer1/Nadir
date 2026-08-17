import DigitCanvas from "@/components/digit-canvas";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
      <h1 className="text-2xl font-semibold">Nadir</h1>
      <DigitCanvas />
    </main>
  );
}
