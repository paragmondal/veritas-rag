"use client";

import React, { useState } from "react";
import { FileText, X } from "lucide-react";

export interface Citation {
  source: string;
  page: number;
  score: number;
  excerpt: string;
}

interface CitationChipProps {
  citation: Citation;
}

export const CitationChip: React.FC<CitationChipProps> = ({ citation }) => {
  const [isOpen, setIsOpen] = useState(false);

  // Shorten filename for clean pill display
  const shortSource = citation.source.replace(".txt", "").replace(".md", "");

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        type="button"
        className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[13px] font-normal text-apple-textLight dark:text-apple-textDark bg-black/[0.04] dark:bg-white/[0.08] hover:bg-black/[0.08] dark:hover:bg-white/[0.12] border border-apple-borderLight dark:border-apple-borderDark rounded-tag transition-transform active:scale-[0.98] cursor-pointer"
        title={`Click to view excerpt from ${citation.source} page ${citation.page}`}
      >
        <FileText className="w-3.5 h-3.5 text-apple-accentLight dark:text-apple-accentDark" />
        <span className="font-semibold">{shortSource}</span>
        <span className="text-apple-subtextLight dark:text-apple-subtextDark">· p.{citation.page}</span>
      </button>

      {/* Excerpt Modal / Popover */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-in fade-in duration-150"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="w-full max-w-lg bg-apple-cardLight dark:bg-apple-cardDark border border-apple-borderLight dark:border-apple-borderDark rounded-card shadow-appleModal p-6 text-left"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between pb-3 border-b border-apple-borderLight dark:border-apple-borderDark mb-4">
              <div>
                <h3 className="text-[17px] font-semibold text-apple-textLight dark:text-apple-textDark flex items-center gap-2">
                  <FileText className="w-4 h-4 text-apple-accentLight dark:text-apple-accentDark" />
                  {citation.source}
                </h3>
                <p className="text-[13px] text-apple-subtextLight dark:text-apple-subtextDark mt-0.5">
                  Page {citation.page} · RRF Fusion Score: {citation.score.toFixed(6)}
                </p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-full hover:bg-black/[0.06] dark:hover:bg-white/[0.1] text-apple-subtextLight dark:text-apple-subtextDark transition-colors"
                aria-label="Close excerpt popover"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="mt-2">
              <h4 className="text-[12px] font-semibold tracking-wide uppercase text-apple-subtextLight dark:text-apple-subtextDark mb-1.5">
                Retrieved Context Excerpt
              </h4>
              <div className="p-3.5 bg-black/[0.02] dark:bg-white/[0.04] border border-apple-borderLight dark:border-apple-borderDark rounded-[12px] text-[14px] leading-relaxed text-apple-textLight dark:text-apple-textDark max-h-60 overflow-y-auto font-sans select-text">
                {citation.excerpt}
              </div>
            </div>

            <div className="mt-5 flex justify-end">
              <button
                onClick={() => setIsOpen(false)}
                className="px-4 py-1.5 text-[14px] font-semibold bg-apple-accentLight dark:bg-apple-accentDark text-white rounded-button hover:opacity-90 active:scale-[0.98] transition-transform"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
