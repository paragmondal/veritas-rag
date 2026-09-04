"use client";

import React from "react";
import { Sliders, Sun, Moon, Database } from "lucide-react";

interface HeaderProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
  onOpenSettings: () => void;
  isBackendHealthy: boolean;
  indexedCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  onToggleDarkMode,
  onOpenSettings,
  isBackendHealthy,
  indexedCount,
}) => {
  return (
    <header className="sticky top-0 z-30 w-full frosted-nav bg-apple-bgLight/80 dark:bg-apple-bgDark/80 border-b border-apple-borderLight dark:border-apple-borderDark transition-colors duration-200">
      <div className="max-w-[760px] mx-auto px-4 h-14 flex items-center justify-between">
        {/* Brand & Title */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-apple-accentLight dark:bg-apple-accentDark flex items-center justify-center text-white shadow-apple">
            <span className="font-semibold text-sm tracking-tight">V</span>
          </div>
          <div>
            <h1 className="text-[17px] font-semibold text-apple-textLight dark:text-apple-textDark tracking-tight leading-none">
              Veritas
            </h1>
            <p className="text-[11px] text-apple-subtextLight dark:text-apple-subtextDark leading-tight mt-0.5">
              Enterprise Hybrid RAG
            </p>
          </div>
        </div>

        {/* Status indicator and action buttons */}
        <div className="flex items-center gap-2">
          {/* Index & Health Status Pill */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-tag text-[12px] bg-black/[0.03] dark:bg-white/[0.05] border border-apple-borderLight dark:border-apple-borderDark">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
              }`}
            />
            <span className="text-apple-subtextLight dark:text-apple-subtextDark">
              {isBackendHealthy ? `${indexedCount} chunks indexed` : "Backend offline"}
            </span>
          </div>

          {/* Dark Mode Toggle */}
          <button
            onClick={onToggleDarkMode}
            type="button"
            className="p-2 rounded-button text-apple-subtextLight dark:text-apple-subtextDark hover:bg-black/[0.05] dark:hover:bg-white/[0.08] active:scale-[0.98] transition-transform"
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Settings Button */}
          <button
            onClick={onOpenSettings}
            type="button"
            className="p-2 rounded-button text-apple-subtextLight dark:text-apple-subtextDark hover:bg-black/[0.05] dark:hover:bg-white/[0.08] active:scale-[0.98] transition-transform"
            aria-label="Open settings"
          >
            <Sliders className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
