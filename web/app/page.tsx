"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { createConversation, listConversations } from "@/lib/api";

/**
 * Entry: land on the newest conversation, creating the first one on a
 * fresh install. Client-side because the browser owns the API origin
 * (`NEXT_PUBLIC_API_URL`) — the web container never proxies the API.
 */
export default function Home() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const conversations = await listConversations();
        const target = conversations[0] ?? (await createConversation());
        if (!cancelled) router.replace(`/c/${target.conversation_id}`);
      } catch (caught) {
        if (!cancelled) setError(String(caught));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  return (
    <div className="flex flex-1 items-center justify-center p-8 text-center">
      {error ? (
        <div className="max-w-md space-y-2">
          <p className="text-sm font-medium">Can&apos;t reach the Varagity API.</p>
          <p className="text-xs text-muted-foreground">
            Start the stack with <code className="rounded bg-muted px-1">docker compose up -d</code> and reload. ({error})
          </p>
        </div>
      ) : (
        <p className="animate-pulse text-sm text-muted-foreground">Loading…</p>
      )}
    </div>
  );
}
