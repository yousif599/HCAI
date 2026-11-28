/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface ShortcutTooltipProps {
  label: string;
  shortcut?: string;
  placement?: 'top' | 'bottom' | 'start' | 'end';
  className?: string;
  children: React.ReactNode;
}

export const ShortcutTooltip: React.FC<ShortcutTooltipProps> = ({
  label,
  shortcut,
  placement = 'top',
  className = 'inline-flex',
  children,
}) => {
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number; ready: boolean }>({
    top: 0,
    left: 0,
    ready: false,
  });
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const openTimerRef = useRef<number | null>(null);
  const tooltipId = useId();

  const clearTimer = useCallback(() => {
    if (openTimerRef.current !== null) {
      window.clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
  }, []);

  const updatePosition = useCallback(() => {
    const triggerEl = triggerRef.current;
    const tipEl = tooltipRef.current;
    if (!triggerEl || !tipEl) return;

    const rect = triggerEl.getBoundingClientRect();
    const tipRect = tipEl.getBoundingClientRect();
    const gap = 8;
    const pad = 8;
    const isRtl = document.documentElement.dir === 'rtl';

    let effectivePlacement = placement;
    if (placement === 'start') {
      effectivePlacement = isRtl ? 'end' : 'start';
    } else if (placement === 'end') {
      effectivePlacement = isRtl ? 'start' : 'end';
    }

    let top = 0;
    let left = 0;

    if (effectivePlacement === 'top') {
      top = rect.top - tipRect.height - gap;
      left = rect.left + rect.width / 2 - tipRect.width / 2;
      if (top < pad) {
        top = rect.bottom + gap;
      }
    } else if (effectivePlacement === 'bottom') {
      top = rect.bottom + gap;
      left = rect.left + rect.width / 2 - tipRect.width / 2;
      if (top + tipRect.height > window.innerHeight - pad) {
        top = rect.top - tipRect.height - gap;
      }
    } else if (effectivePlacement === 'end') {
      top = rect.top + rect.height / 2 - tipRect.height / 2;
      left = rect.right + gap;
      if (left + tipRect.width > window.innerWidth - pad) {
        left = rect.left - tipRect.width - gap;
      }
    } else {
      top = rect.top + rect.height / 2 - tipRect.height / 2;
      left = rect.left - tipRect.width - gap;
      if (left < pad) {
        left = rect.right + gap;
      }
    }

    left = Math.max(pad, Math.min(left, window.innerWidth - tipRect.width - pad));
    top = Math.max(pad, Math.min(top, window.innerHeight - tipRect.height - pad));

    setCoords({ top, left, ready: true });
  }, [placement]);

  useEffect(() => {
    if (!visible) {
      setCoords((prev) => (prev.ready ? { ...prev, ready: false } : prev));
      return;
    }

    const rafId = window.requestAnimationFrame(updatePosition);
    const hideOnScroll = () => setVisible(false);
    window.addEventListener('scroll', hideOnScroll, true);
    window.addEventListener('resize', updatePosition);
    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener('scroll', hideOnScroll, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [visible, updatePosition, label, shortcut]);

  const handleClose = useCallback(() => {
    clearTimer();
    setVisible(false);
  }, [clearTimer]);

  const handleOpen = useCallback(() => {
    clearTimer();
    openTimerRef.current = window.setTimeout(() => {
      setVisible(true);
    }, 1000);
  }, [clearTimer]);

  // Dismiss or cancel pending tooltip immediately whenever the user presses any key/shortcut
  useEffect(() => {
    const onGlobalKeyDown = () => {
      handleClose();
    };
    window.addEventListener('keydown', onGlobalKeyDown, true);
    return () => {
      window.removeEventListener('keydown', onGlobalKeyDown, true);
    };
  }, [handleClose]);

  useEffect(() => clearTimer, [clearTimer]);

  return (
    <>
      <span
        ref={triggerRef}
        className={className}
        aria-describedby={visible ? tooltipId : undefined}
        onMouseEnter={handleOpen}
        onMouseLeave={handleClose}
        onBlur={handleClose}
        onPointerDown={handleClose}
      >
        {children}
      </span>

      {visible &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            ref={tooltipRef}
            id={tooltipId}
            role="tooltip"
            style={{
              position: 'fixed',
              top: `${coords.top}px`,
              left: `${coords.left}px`,
              opacity: coords.ready ? 1 : 0,
            }}
            className="pointer-events-none z-[100] flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900/95 dark:bg-slate-800/95 text-white text-[0.72rem] font-medium shadow-lg border border-slate-700/70 backdrop-blur-xs whitespace-nowrap transition-opacity duration-100 select-none"
          >
            <span>{label}</span>
            {shortcut && (
              <kbd
                dir="ltr"
                className="px-1.5 py-0.5 rounded bg-white/15 border border-white/20 text-[0.65rem] font-mono font-semibold tracking-wide text-emerald-200 leading-none"
              >
                {shortcut}
              </kbd>
            )}
          </div>,
          document.body
        )}
    </>
  );
};
