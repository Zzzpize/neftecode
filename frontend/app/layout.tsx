import type { Metadata } from 'next';
import { Providers } from '@/components/Providers';
import { TimeProvider } from '@/components/TimeStore';
import './globals.css';

export const metadata: Metadata = {
  title: 'Neftecode',
  description: 'МАС-советник для оператора НПЗ',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru">
      <body className="bg-neutral-950 text-neutral-100 antialiased">
        <Providers>
          <TimeProvider>{children}</TimeProvider>
        </Providers>
      </body>
    </html>
  );
}
