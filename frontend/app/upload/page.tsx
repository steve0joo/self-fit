'use client';

import React, { useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setFileName(file.name);
  };

  const startAnalysis = () => {
    if (!fileName) return;
    setAnalyzing(true);
    setProgress(0);
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          clearInterval(interval);
          setTimeout(() => router.push('/interview/report'), 400);
          return 100;
        }
        return p + 8;
      });
    }, 200);
  };

  return (
    <div className="page-shell">
      <Link className="page-back" href="/">← 홈으로</Link>
      <h1 className="page-title">영상 업로드 분석</h1>
      <p className="page-sub" style={{ color: 'var(--ink)' }}>이미 녹화한 면접 영상을 올리면 동일한 안구, 표정 모델로 분석합니다.</p>

      <div className="upload-box" onClick={() => inputRef.current?.click()}>
        <div style={{ fontSize: 28 }}>🎬</div>
        <p>{fileName ? fileName : 'MP4, MOV 파일을 선택하거나 끌어다 놓으세요'}</p>
        <input ref={inputRef} type="file" accept="video/*" onChange={handleFile} style={{ display: 'none' }} />
      </div>

      {analyzing && (
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      <div className="action-row">
        <button className="btn-solid" disabled={!fileName || analyzing} onClick={startAnalysis}>
          {analyzing ? `분석 중… ${progress}%` : '분석 시작 →'}
        </button>
      </div>
    </div>
  );
}