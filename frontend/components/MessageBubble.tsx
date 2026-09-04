"use client";

import React from "react";
import { Citation, CitationChip } from "./CitationChip";
import { Zap, AlertCircle } from "lucide-react";

export interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  citations?: Citation[];
  latencySeconds?: number;
  isError?: boolean;
}

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === "user";

  // Frontend safety deduplication by (source, page)
  const uniqueCitations = React.useMemo(() => {
    if (!message.citations || message.citations.length === 0) return [];
    const seen = new Set<string>();
    const deduped: Citation[] = [];
    for (const c of message.citations) {
      const key = `${c.source}_p${c.page}`;
      if (!seen.has(key)) {
        seen.add(key);
        deduped.push(c);
      }
    }
    return deduped;
  }, [message.citations]);

  return (
    <div
      className={`flex flex-col w-full my-3 animate-in fade-in duration-200 ${
        isUser ? "items-end" : "items-start"
      }`}
    >
      {/* Bubble Container */}
      <div
        className={`max-w-[85%] sm:max-w-[75%] px-4 py-3 text-[15px] leading-[1.5] ${
          isUser
            ? "bg-apple-userBubble text-white rounded-[18px] rounded-br-[4px] shadow-sm"
            : message.isError
            ? "bg-rose-50 dark:bg-rose-950/40 text-rose-800 dark:text-rose-200 border border-rose-200 dark:border-rose-900 rounded-[18px] rounded-bl-[4px]"
            : "bg-apple-cardLight dark:bg-apple-cardDark text-apple-textLight dark:text-apple-textDark border border-apple-borderLight dark:border-apple-borderDark rounded-[18px] rounded-bl-[4px] shadow-apple"
        }`}
      >
        {/* Message Text with Paragraphs */}
        <div className="whitespace-pre-wrap select-text">
          {message.text}
        </div>

        {/* Latency and provenance metadata */}
        {!isUser && message.latencySeconds !== undefined && (
          <div className="mt-2.5 pt-2 border-t border-apple-borderLight/60 dark:border-apple-borderDark/60 flex items-center justify-between text-[11px] text-apple-subtextLight dark:text-apple-subtextDark">
            <span className="flex items-center gap-1 font-medium">
              <Zap className="w-3 h-3 text-amber-500" />
              {(message.latencySeconds * 1000).toFixed(0)}ms latency
            </span>
            <span>RRF Hybrid Retrieval</span>
          </div>
        )}
      </div>

      {/* Deduplicated Citations List below assistant response */}
      {!isUser && uniqueCitations.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 max-w-[85%] sm:max-w-[75%]">
          {uniqueCitations.map((citation, idx) => (
            <CitationChip
              key={`${citation.source}-${citation.page}-${idx}`}
              citation={citation}
            />
          ))}
        </div>
      )}
    </div>
  );
};
