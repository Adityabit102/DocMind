"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Confidence, SentenceSupport, SourceChunk } from "./types";

/**
 * Local, persistent chat history. Conversations live in
 * localStorage so they survive reloads and you can revisit past chats; nothing
 * is lost until you delete a chat. The hook exposes a `turns`/`setTurns` pair
 * scoped to the active conversation, so the chat page barely changes, plus
 * new/select/delete operations over the full list.
 */
export interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  confidence?: Confidence;
  grounding?: number | null;
  spans?: SentenceSupport[];
  model?: string;
  tokens?: number;
  cost?: number;
  streaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  turns: Turn[];
  updatedAt: number;
}

const LS_CHATS = "docmind.chats.v1";
const LS_ACTIVE = "docmind.activeChat.v1";

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

function deriveTitle(turns: Turn[]): string {
  const first = turns.find((t) => t.role === "user");
  if (!first) return "New chat";
  const t = first.content.trim();
  return t.length > 48 ? `${t.slice(0, 48)}…` : t || "New chat";
}

const emptyChat = (): Conversation => ({
  id: uid(),
  title: "New chat",
  turns: [],
  updatedAt: Date.now(),
});

export function useChats() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const loaded = useRef(false);

  // Hydrate from localStorage once (client only).
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_CHATS);
      const list: Conversation[] = raw ? JSON.parse(raw) : [];
      // Clear any transient streaming flags left by an interrupted generation.
      list.forEach((c) => c.turns.forEach((t) => (t.streaming = false)));
      const active = localStorage.getItem(LS_ACTIVE) ?? "";
      if (list.length === 0) {
        const c = emptyChat();
        setConversations([c]);
        setActiveId(c.id);
      } else {
        setConversations(list);
        setActiveId(list.some((c) => c.id === active) ? active : list[0].id);
      }
    } catch {
      const c = emptyChat();
      setConversations([c]);
      setActiveId(c.id);
    }
    loaded.current = true;
  }, []);

  // Persist (debounced) so streaming token writes don't thrash localStorage.
  useEffect(() => {
    if (!loaded.current) return;
    const id = setTimeout(() => {
      try {
        localStorage.setItem(LS_CHATS, JSON.stringify(conversations));
        localStorage.setItem(LS_ACTIVE, activeId);
      } catch {
        /* quota / unavailable — ignore */
      }
    }, 400);
    return () => clearTimeout(id);
  }, [conversations, activeId]);

  const turns = conversations.find((c) => c.id === activeId)?.turns ?? [];

  const setTurns = useCallback(
    (updater: Turn[] | ((t: Turn[]) => Turn[])) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const next = typeof updater === "function" ? updater(c.turns) : updater;
          return {
            ...c,
            turns: next,
            title: c.title === "New chat" ? deriveTitle(next) : c.title,
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [activeId],
  );

  const newChat = useCallback(() => {
    setConversations((prev) => {
      const existingEmpty = prev.find((c) => c.turns.length === 0);
      if (existingEmpty) {
        setActiveId(existingEmpty.id);
        return prev;
      }
      const c = emptyChat();
      setActiveId(c.id);
      return [c, ...prev];
    });
  }, []);

  const selectChat = useCallback((id: string) => setActiveId(id), []);

  const deleteChat = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (next.length === 0) {
          const c = emptyChat();
          setActiveId(c.id);
          return [c];
        }
        if (id === activeId) setActiveId(next[0].id);
        return next;
      });
    },
    [activeId],
  );

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  return { conversations: sorted, activeId, turns, setTurns, newChat, selectChat, deleteChat };
}
