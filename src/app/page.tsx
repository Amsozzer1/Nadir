import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-4">
      <h1 className="text-2xl font-semibold">Nadir</h1>
      <p className="text-muted-foreground text-sm">
        Next.js + Tailwind + shadcn/ui + Firebase
      </p>
      <Button>Click me</Button>
    </main>
  );
}
