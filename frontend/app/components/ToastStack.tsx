'use client';

import React from 'react';

export interface ToastItem {
  id: number;
  icon: string;
  message: string;
}

const ToastStack: React.FC<{ toasts: ToastItem[] }> = ({ toasts }) => {
  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <div className="toast" key={t.id}>
          <span>{t.icon}</span>
          {t.message}
        </div>
      ))}
    </div>
  );
};

export default ToastStack;