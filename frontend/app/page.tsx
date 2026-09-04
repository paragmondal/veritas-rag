"use client";

import React, { useState, useEffect } from "react";
import { Header } from "../components/Header";
import { ChatWindow } from "../components/ChatWindow";
import { SettingsSheet } from "../components/SettingsSheet";
import { Message } from "../components/MessageBubble";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [darkMode, setDarkMode] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // System settings
  const [embeddingBackend, setEmbeddingBackend] = useState("tfidf");
  const [llmProvider, setLlmProvider] = useState("mock");
  const [topK, setTopK] = useState(5);

  // Health and index status
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [indexedCount, setIndexedCount] = useState(0);

  // Initialize theme based on prefers-color-scheme
  useEffect(() => {
    if (typeof window !== "undefined") {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      setDarkMode(prefersDark);
      if (prefersDark) {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
    }
  }, []);

  const handleToggleDarkMode = () => {
    const nextMode = !darkMode;
    setDarkMode(nextMode);
    if (nextMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Check health on mount
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setIsBackendHealthy(data.status === "healthy");
        setIndexedCount(data.chunks_count || 0);
        if (data.embedding_backend) setEmbeddingBackend(data.embedding_backend);
        if (data.llm_provider) setLlmProvider(data.llm_provider);
      } else {
        setIsBackendHealthy(false);
      }
    } catch {
      setIsBackendHealthy(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleSendMessage = async (query: string) => {
    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `asst-${Date.now()}`;

    const userMessage: Message = {
      id: userMessageId,
      sender: "user",
      text: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: query,
          top_k: topK,
          embedding_backend: embeddingBackend,
          provider: llmProvider,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();

      const assistantMessage: Message = {
        id: assistantMessageId,
        sender: "assistant",
        text: data.answer,
        citations: data.citations || [],
        latencySeconds: data.latency_seconds,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        id: assistantMessageId,
        sender: "assistant",
        text: `Unable to process query: ${err.message || "Failed to reach Veritas backend."}`,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReindex = async () => {
    const res = await fetch(`${API_BASE_URL}/reindex`, {
      method: "POST",
    });
    if (res.ok) {
      const data = await res.json();
      setIndexedCount(data.chunks_count || 0);
      setIsBackendHealthy(true);
    } else {
      throw new Error("Failed to re-index");
    }
  };

  return (
    <main className="min-h-screen flex flex-col bg-apple-bgLight dark:bg-apple-bgDark transition-colors duration-200">
      <Header
        darkMode={darkMode}
        onToggleDarkMode={handleToggleDarkMode}
        onOpenSettings={() => setIsSettingsOpen(true)}
        isBackendHealthy={isBackendHealthy}
        indexedCount={indexedCount}
      />

      <ChatWindow
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />

      <SettingsSheet
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        embeddingBackend={embeddingBackend}
        onChangeEmbeddingBackend={setEmbeddingBackend}
        llmProvider={llmProvider}
        onChangeLlmProvider={setLlmProvider}
        topK={topK}
        onChangeTopK={setTopK}
        onReindex={handleReindex}
      />
    </main>
  );
}
