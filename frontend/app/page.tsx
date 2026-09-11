import { StatePanel } from '@/components/StatePanel';
import { RecommendationCard } from '@/components/RecommendationCard';
import { AgentTrace } from '@/components/AgentTrace';
import { TimeMachine } from '@/components/TimeMachine';
import { TagChart } from '@/components/TagChart';
import { WeightSliders } from '@/components/WeightSliders';
import { ParetoChart } from '@/components/ParetoChart';
import { QAChat } from '@/components/QAChat';

export default function Home() {
  return (
    <main className="min-h-screen p-6">
      <header className="mb-6 flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Neftecode</h1>
          <p className="text-sm text-neutral-400">
            МАС-советник · АВТ → гидроочистка → блендинг
          </p>
        </div>
        <TimeMachine />
      </header>

      <div className="grid grid-cols-12 gap-4">
        <section className="col-span-3 space-y-4">
          <StatePanel />
          <WeightSliders />
        </section>

        <section className="col-span-6 space-y-4">
          <RecommendationCard />
          <ParetoChart />
          <div className="grid grid-cols-2 gap-4">
            <TagChart tag="pak_sulfur_ppm" />
            <TagChart tag="avt_T55" />
          </div>
        </section>

        <section className="col-span-3 space-y-4">
          <QAChat />
          <AgentTrace />
        </section>
      </div>
    </main>
  );
}
