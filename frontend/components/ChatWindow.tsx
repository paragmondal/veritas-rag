"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Sparkles } from "lucide-react";
import { Message, MessageBubble } from "./MessageBubble";
import { Citation } from "./CitationChip";

interface ChatWindowProps {
  messages: Message[];
  onSendMessage: (query: string) => Promise<void>;
  isLoading: boolean;
}

const SUGGESTED_QUERIES = [
  "How did revenue change from 2024 to 2025?",
  "What is Acme's exposure to supply chain disruption in 2025?",
  "Compare gross margins between FY2024 and FY2025.",
  "What are the quorum requirements for the Audit Committee?",
  "What are Acme's planned investments in lunar quantum computers for 2035?",
];

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  onSendMessage,
  isLoading,
}) => {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages or loading change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Adjust textarea height dynamically
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        140
      )}px`;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || isLoading) return;

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    await onSendMessage(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col max-w-[760px] w-full mx-auto px-4 justify-between h-[calc(100vh-3.5rem)]">
      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto py-6 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <div className="w-12 h-12 rounded-full bg-apple-accentLight/10 dark:bg-apple-accentDark/15 flex items-center justify-center mb-4 text-apple-accentLight dark:text-apple-accentDark">
              <Sparkles className="w-6 h-6" />
            </div>
            <h2 className="text-[22px] font-semibold text-apple-textLight dark:text-apple-textDark tracking-tight mb-2">
              Corporate Intelligence with Veritas
            </h2>
            <p className="text-[15px] text-apple-subtextLight dark:text-apple-subtextDark max-w-md mb-8 leading-relaxed">
              Synthesizing multi-year 10-K filings, risk factors, and governance charters using dense + BM25 hybrid search.
            </p>

            {/* Suggested Starter Prompts */}
            <div className="w-full max-w-md space-y-2">
              <p className="text-[12px] font-semibold uppercase tracking-wider text-apple-subtextLight dark:text-apple-subtextDark mb-1 text-left">
                Suggested Inquiries
              </p>
              {SUGGESTED_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(q)}
                  type="button"
                  className="w-full text-left px-3.5 py-2.5 rounded-card bg-apple-cardLight dark:bg-apple-cardDark border border-apple-borderLight dark:border-apple-borderDark text-[13px] text-apple-textLight dark:text-apple-textDark hover:border-apple-accentLight dark:hover:border-apple-accentDark shadow-apple transition-colors active:scale-[0.99]"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} />)
        )}

        {/* Shimmer Skeleton Loading State */}
        {isLoading && (
          <div className="flex flex-col items-start w-full my-3">
            <div className="max-w-[85%] sm:max-w-[75%] p-4 bg-apple-cardLight dark:bg-apple-cardDark border border-apple-borderLight dark:border-apple-borderDark rounded-[18px] rounded-bl-[4px] shadow-apple w-full space-y-2.5">
              <div className="h-4 w-3/4 rounded-md shimmer-loading" />
              <div className="h-4 w-full rounded-md shimmer-loading" />
              <div className="h-4 w-5/6 rounded-md shimmer-loading" />
              <div className="h-3 w-1/3 rounded-md shimmer-loading pt-2" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Dock */}
      <div className="pb-6 pt-2 sticky bottom-0 bg-gradient-to-t from-apple-bgLight via-apple-bgLight/90 to-transparent dark:from-apple-bgDark dark:via-apple-bgDark/90">
        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2 bg-apple-cardLight dark:bg-apple-cardDark border border-apple-borderLight dark:border-apple-borderDark rounded-[20px] p-1.5 shadow-apple focus-within:ring-2 focus-within:ring-apple-accentLight dark:focus-within:ring-apple-accentDark transition-all"
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about Acme's 10-K filings or charter..."
            className="flex-1 bg-transparent px-3 py-2 text-[15px] leading-relaxed text-apple-textLight dark:text-apple-textDark placeholder:text-apple-subtextLight dark:placeholder:text-apple-subtextDark focus:outline-none resize-none max-h-36"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="w-9 h-9 flex items-center justify-center rounded-full bg-apple-accentLight dark:bg-apple-accentDark text-white disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-90 active:scale-[0.95] transition-transform flex-shrink-0"
            aria-label="Send message"
          >
            <ArrowUp className="w-5 h-5" />
          </button>
        </form>
        <div className="text-center mt-2 text-[11px] text-apple-subtextLight dark:text-apple-subtextDark">
          Veritas cites verified [source, page] passages only · Press Return to submit
        </div>
      </div>
    </div>
  );
};
