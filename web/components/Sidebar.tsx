import { SidebarContent } from "@/components/shell/SidebarContent";

/**
 * The persistent desktop (`md:`) left rail. Below `md` the same content is
 * served by the `MobileTopBar` navigation drawer instead.
 */
export function Sidebar() {
  return (
    <aside className="hidden h-full w-64 shrink-0 flex-col border-r border-border bg-sidebar text-sidebar-foreground md:flex">
      <SidebarContent />
    </aside>
  );
}
