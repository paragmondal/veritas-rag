"use client";

import React, { useState } from "react";
import { X, RefreshCw, CheckCircle2 } from "lucide-react";

interface SettingsSheetProps {
  isOpen: boolean;
  onClose: () => void;
  embeddingBackend: string;
  onChangeEmbeddingBackend: (val: string) => void;
  llmProvider: string;
  onChangeLlmProvider: (val: string) => void;
  topK: number;
  onChangeTopK: (val: number) => void;
  onReindex: () => Promise<void>;
}

export const SettingsSheet: React.FC<SettingsSheetProps> = ({
  isOpen,
  onClose,
  embeddingBackend,
  onChangeEmbeddingBackend,
  llmProvider,
  onChangeLlmProvider,
  topK,
  onChangeTopK,
  onReindex,
}) => {
  const [reindexing, setReindexing] = useState(false);
  const [reindexSuccess, setReindexSuccess] = useState(false);

  if (!isOpen) return null;

  const handleReindex = async () => {
    setReindexing(true);
    setReindexSuccess(false);
    try {
      await onReindex();
      setReindexSuccess(true);
      setTimeout(() => setReindexSuccess(false), 3000);
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm h-full bg-apple-cardLight dark:bg-apple-cardDark border-l border-apple-borderLight dark:border-apple-borderDark shadow-2xl p-6 flex flex-col justify-between"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-apple-borderLight dark:border-apple-borderDark mb-6">
            <h2 className="text-[18px] font-semibold text-apple-textLight dark:text-apple-textDark">
              Pipeline Settings
            </h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-full hover:bg-black/[0.06] dark:hover:bg-white/[0.1] text-apple-subtextLight dark:text-apple-subtextDark transition-colors"
              aria-label="Close settings"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="space-y-6">
            {/* Embedding Backend */}
            <div>
              <label className="block text-[13px] font-semibold text-apple-textLight dark:text-apple-textDark mb-1.5">
                Embedding Backend
              </label>
              <p className="text-[12px] text-apple-subtextLight dark:text-apple-subtextDark mb-2">
                Select dense vector generation backend.
              </p>
              <select
                value={embeddingBackend}
                onChange={(e) => onChangeEmbeddingBackend(e.target.value)}
                className="w-full px-3 py-2 text-[14px] rounded-button bg-black/[0.03] dark:bg-white/[0.06] border border-apple-borderLight dark:border-apple-borderDark text-apple-textLight dark:text-apple-textDark focus:outline-none focus:ring-2 focus:ring-apple-accentLight dark:focus:ring-apple-accentDark"
              >
                <option value="tfidf">TF-IDF (Offline, Zero-key Default)</option>
                <option value="openai">OpenAI text-embedding-3-small</option>
              </select>
            </div>

            {/* LLM Provider */}
            <div>
              <label className="block text-[13px] font-semibold text-apple-textLight dark:text-apple-textDark mb-1.5">
                Generation Provider
              </label>
              <p className="text-[12px] text-apple-subtextLight dark:text-apple-subtextDark mb-2">
                Select answer generation model.
              </p>
              <select
                value={llmProvider}
                onChange={(e) => onChangeLlmProvider(e.target.value)}
                className="w-full px-3 py-2 text-[14px] rounded-button bg-black/[0.03] dark:bg-white/[0.06] border border-apple-borderLight dark:border-apple-borderDark text-apple-textLight dark:text-apple-textDark focus:outline-none focus:ring-2 focus:ring-apple-accentLight dark:focus:ring-apple-accentDark"
              >
                <option value="mock">Mock Extractive (Offline Default)</option>
                <option value="openai">OpenAI Chat Completion</option>
                <option value="anthropic">Anthropic Claude Messages</option>
              </select>
            </div>

            {/* Retrieval Final Top-K */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-[13px] font-semibold text-apple-textLight dark:text-apple-textDark">
                  Retrieved Passages (top_k)
                </label>
                <span className="text-[13px] font-semibold text-apple-accentLight dark:text-apple-accentDark">
                  {topK}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="15"
                value={topK}
                onChange={(e) => onChangeTopK(Number(e.target.value))}
                className="w-full accent-apple-accentLight dark:accent-apple-accentDark cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-apple-subtextLight dark:text-apple-subtextDark mt-1">
                <span>1 chunk</span>
                <span>15 chunks</span>
              </div>
            </div>

            {/* Corpus Re-indexing */}
            <div className="pt-4 border-t border-apple-borderLight dark:border-apple-borderDark">
              <label className="block text-[13px] font-semibold text-apple-textLight dark:text-apple-textDark mb-1.5">
                Index Management
              </label>
              <p className="text-[12px] text-apple-subtextLight dark:text-apple-subtextDark mb-3">
                Re-process raw files and rebuild Chroma and BM25 indexes.
              </p>
              <button
                onClick={handleReindex}
                disabled={reindexing}
                type="button"
                className="w-full py-2.5 px-3 flex items-center justify-center gap-2 rounded-button text-[14px] font-semibold bg-black/[0.05] dark:bg-white/[0.08] hover:bg-black/[0.08] dark:hover:bg-white/[0.12] active:scale-[0.98] transition-transform text-apple-textLight dark:text-apple-textDark border border-apple-borderLight dark:border-apple-borderDark disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${reindexing ? "animate-spin" : ""}`} />
                {reindexing ? "Re-indexing Corpus..." : "Re-index Corpus"}
              </button>
              {reindexSuccess && (
                <p className="mt-2 text-[12px] text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Re-indexing complete!
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-apple-borderLight dark:border-apple-borderDark text-center">
          <p className="text-[11px] text-apple-subtextLight dark:text-apple-subtextDark">
            Veritas Architecture: Chroma Dense + BM25 Sparse with RRF (k=60)
          </p>
        </div>
      </div>
    </div>
  );
};
