"use client";

import { RotateCcwIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { createConversation, listConversations } from "@/lib/api";
import { describeChatError } from "@/lib/errors";

/**
 * Entry: land on the newest conversation, creating the first one on a
 * fresh install. Client-side because the browser owns the API origin
 * (`NEXT_PUBLIC_API_URL`) — the web container never proxies the API.
 */
export default function Home() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0); // bump to retry the bootstrap

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
  }, [router, attempt]);

  if (error) {
    // The same banner language the chat's error path speaks (lib/errors).
    const descriptor = describeChatError({
      code: "network_error",
      message: error,
    });
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div
          role="alert"
          className="w-fit max-w-md space-y-1.5 rounded-lg border border-destructive/25 bg-destructive/5 p-4 text-xs"
        >
          <p className="text-sm font-medium text-destructive">
            {descriptor.title}
          </p>
          {descriptor.hint && (
            <p className="text-muted-foreground">{descriptor.hint}</p>
          )}
          {descriptor.command && (
            <code className="block w-fit rounded-md border border-border/60 bg-muted px-1.5 py-0.5 font-mono">
              {descriptor.command}
            </code>
          )}
          <p className="font-mono break-words text-muted-foreground">{error}</p>
          <div className="pt-1">
            <Button
              size="xs"
              variant="outline"
              onClick={() => {
                setError(null);
                setAttempt((current) => current + 1);
              }}
            >
              <RotateCcwIcon aria-hidden />
              Try again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      role="status"
      aria-busy="true"
      aria-label="Loading"
      className="flex flex-1 items-center justify-center p-8"
    >
      <div className="flex w-full max-w-sm flex-col gap-3">
        <Skeleton className="h-4 w-3/5" />
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-4 w-2/5" />
      </div>
    </div>
  );
}
