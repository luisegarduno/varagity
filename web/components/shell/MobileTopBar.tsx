"use client";

import { MenuIcon, PlusIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { SidebarContent } from "@/components/shell/SidebarContent";
import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerContent,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { createConversation } from "@/lib/api";
import { notifyConversationsChanged } from "@/lib/conversations-bus";

/**
 * The `<md` shell: a slim top bar (hamburger → left navigation drawer,
 * wordmark, "new chat") rendered above every route by the root layout.
 * The drawer hosts the same `SidebarContent` as the desktop rail and
 * closes itself before any navigation via `onNavigate`.
 *
 * `contentClassName="p-0"` zeroes `DrawerContent`'s default padding so the
 * sidebar's own paddings land flush against the panel edges, while keeping
 * `Drawer.Content` in the tree (text selection without swipe interference).
 */
export function MobileTopBar() {
  const router = useRouter();
  const [open, setOpen] = useState(false);

  async function handleNewChat() {
    try {
      const created = await createConversation();
      notifyConversationsChanged();
      router.push(`/c/${created.conversation_id}`);
    } catch {
      setOpen(true); // the drawer's list surfaces the "API unreachable" state
    }
  }

  return (
    <header className="flex h-12 shrink-0 items-center gap-1 border-b border-border bg-background/85 px-2 backdrop-blur md:hidden">
      <Drawer side="left" open={open} onOpenChange={setOpen}>
        <DrawerTrigger
          render={
            <Button variant="ghost" size="icon-sm" aria-label="Open navigation" />
          }
        >
          <MenuIcon />
        </DrawerTrigger>
        <DrawerContent
          className="bg-sidebar text-sidebar-foreground"
          contentClassName="min-h-0 gap-0 p-0"
        >
          <DrawerTitle className="sr-only">Navigation</DrawerTitle>
          <SidebarContent onNavigate={() => setOpen(false)} />
        </DrawerContent>
      </Drawer>

      <span className="flex-1 text-center font-heading text-lg leading-none font-normal italic">
        Varagity
      </span>

      <Button
        variant="ghost"
        size="icon-sm"
        aria-label="New chat"
        onClick={() => void handleNewChat()}
      >
        <PlusIcon />
      </Button>
    </header>
  );
}
