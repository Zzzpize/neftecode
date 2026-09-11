'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { api } from '@/lib/api';

type Message = { role: 'user' | 'assistant'; text: string };

const SUGGESTIONS = [
  'А что если ничего не менять?',
  'Почему именно эти теги?',
  'Насколько ты уверен в прогнозе?',
];

export function QAChat() {
  const { decisionId } = useTime();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);

  const ask = useMutation({
    mutationFn: async (question: string) => {
      if (!decisionId) throw new Error('нет активной рекомендации');
      return api.ask(decisionId, question);
    },
    onSuccess: (data) => {
      setMessages((m) => [...m, { role: 'assistant', text: data.answer }]);
    },
    onError: (err: Error) => {
      setMessages((m) => [...m, { role: 'assistant', text: `Ошибка: ${err.message}` }]);
    },
  });

  const send = (text: string) => {
    if (!text.trim() || ask.isPending) return;
    setMessages((m) => [...m, { role: 'user', text }]);
    setInput('');
    ask.mutate(text);
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-neutral-400">
          Вопрос оператора
        </span>
        <span className="text-xs text-neutral-500">GigaChat</span>
      </h2>

      {messages.length === 0 && (
        <div className="mb-3 space-y-1">
          <div className="text-xs text-neutral-500 mb-2">Быстрые вопросы:</div>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              disabled={!decisionId || ask.isPending}
              className="block w-full text-left text-xs rounded border border-neutral-800 bg-neutral-950 px-2 py-1.5 text-neutral-400 hover:border-neutral-700 hover:text-neutral-200 disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {messages.length > 0 && (
        <div className="mb-3 space-y-2 max-h-64 overflow-y-auto">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`rounded px-2 py-1.5 text-xs ${
                m.role === 'user'
                  ? 'bg-cyan-950/40 border border-cyan-800/40 text-neutral-200'
                  : 'bg-neutral-950 border border-neutral-800 text-neutral-300'
              }`}
            >
              <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-0.5">
                {m.role === 'user' ? 'вы' : 'ассистент'}
              </div>
              {m.text}
            </div>
          ))}
          {ask.isPending && (
            <div className="rounded bg-neutral-950 border border-neutral-800 px-2 py-1.5 text-xs text-neutral-500">
              <span className="inline-block animate-pulse">думает...</span>
            </div>
          )}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={decisionId ? 'Ваш вопрос...' : 'Ждём рекомендацию...'}
          disabled={!decisionId || ask.isPending}
          className="flex-1 bg-neutral-950 border border-neutral-800 rounded px-2 py-1.5 text-xs text-neutral-100 placeholder:text-neutral-600 focus:outline-none focus:border-neutral-600 disabled:opacity-40"
        />
        <button
          type="submit"
          disabled={!decisionId || ask.isPending || !input.trim()}
          className="rounded border border-neutral-800 bg-neutral-950 px-3 py-1.5 text-xs text-neutral-300 hover:border-neutral-700 hover:text-neutral-100 disabled:opacity-40"
        >
          Спросить
        </button>
      </form>

      {!decisionId && messages.length === 0 && (
        <div className="mt-2 text-[11px] text-neutral-500">
          Выберите момент времени, чтобы получить активную рекомендацию.
        </div>
      )}
    </div>
  );
}
