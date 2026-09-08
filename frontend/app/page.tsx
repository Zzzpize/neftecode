import { RecommendationCard } from '@/components/RecommendationCard';
import { StatePanel } from '@/components/StatePanel';
import { AgentTrace } from '@/components/AgentTrace';
import { TimeMachine } from '@/components/TimeMachine';
import { QAChat } from '@/components/QAChat';

export default function Home() {
  return (
    <main className="min-h-screen p-6">
      <header className="mb-6 flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Neftecode</h1>
          <p className="text-sm text-neutral-400">
            МАС-советник
          </p>
        </div>
        <TimeMachine />
      </header>

      <div className="grid grid-cols-12 gap-4">
        <section className="col-span-3">
          <StatePanel />
        </section>
        <section className="col-span-6 space-y-4">
          <RecommendationCard />
          <QAChat />
        </section>
        <section className="col-span-3">
          <AgentTrace />
        </section>
      </div>
    </main>
  );
}
