import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SelfFit — AI 면접 자가진단',
  description: '웹캠으로 시선과 표정을 실시간 분석하고, 면접이 끝나면 행동 리포트를 제공합니다.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <div className="selffit-app">{children}</div>
      </body>
    </html>
  );
}