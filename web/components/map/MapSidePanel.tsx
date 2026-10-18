/**
 * The floating side panel over the map canvas: project name + tagline, then
 * the Models / Tools / Integrations callouts with their vendored favicons —
 * foglamp's left card, restyled onto the house tokens. A tap folds it down to
 * a title pill (on a phone it otherwise covers most of the canvas); another
 * tap opens it again.
 */

"use client";

import { ChevronDownIcon } from "lucide-react";
import { useId, useState } from "react";

import type { CodebaseMap, TopItem } from "@/lib/codebase-map";
import { cn } from "@/lib/utils";

import { useIconSrc } from "./icon-base";
import { Glyph, type GLYPH_PATHS } from "./MapNode";

/** One callout row: favicon (with glyph fallback) + label. */
function TopRow({
  item,
  fallback,
}: {
  item: TopItem;
  fallback: keyof typeof GLYPH_PATHS;
}) {
  const [failed, setFailed] = useState(false);
  const iconSrc = useIconSrc();
  const domain = item.domain;
  return (
    <li className="flex items-center gap-2">
      {domain && !failed ? (
        // Static vendored asset; next/image adds nothing for a 14px favicon.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={iconSrc(domain)}
          alt=""
          onError={() => setFailed(true)}
          className="size-3.5 rounded-[4px] object-contain"
        />
      ) : (
        <Glyph name={fallback} className="size-3.5 text-muted-foreground" />
      )}
      <span className="truncate text-sm font-medium">{item.label}</span>
    </li>
  );
}

/** One labeled callout section; renders nothing for an empty list. */
function TopSection({
  title,
  icon,
  items,
  fallback,
  className,
}: {
  title: string;
  icon: keyof typeof GLYPH_PATHS;
  items: readonly TopItem[];
  fallback: keyof typeof GLYPH_PATHS;
  className?: string;
}) {
  if (items.length === 0) return null;
  return (
    <section className={className}>
      <h2 className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
        <Glyph name={icon} className="size-3.5" />
        <span className="leading-none">{title}</span>
      </h2>
      <ul className="flex list-none flex-col gap-3">
        {items.map((item) => (
          <TopRow key={item.id} item={item} fallback={fallback} />
        ))}
      </ul>
    </section>
  );
}

/**
 * The panel card, absolutely positioned over the canvas by the view. The whole
 * card is the tap target; the title button carries the disclosure semantics
 * for keyboard and screen-reader users and has no handler of its own — its
 * click (pointer, Enter, or Space) bubbles to the card's, so it toggles once.
 */
export function MapSidePanel({ map }: { map: CodebaseMap }) {
  const [open, setOpen] = useState(true);
  const bodyId = useId();
  return (
    <div
      onClick={() => setOpen((prev) => !prev)}
      className={cn(
        "pointer-events-auto absolute top-4 left-4 z-20 flex max-h-[70%] cursor-pointer flex-col overflow-y-auto border border-border bg-card/95 text-card-foreground backdrop-blur-sm [box-shadow:var(--map-card-shadow)] dark:border-transparent",
        open ? "w-60 rounded-[28px] p-5 sm:w-64" : "rounded-full py-2.5 pr-4 pl-5",
      )}
    >
      <h1 className="font-heading text-xl leading-tight font-normal">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          className="flex w-full cursor-pointer items-center justify-between gap-3 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
        >
          <span>
            {map.project.name}
            <span className="sr-only"> codebase map</span>
          </span>
          <ChevronDownIcon
            aria-hidden
            className={cn(
              "size-4 shrink-0 text-muted-foreground motion-safe:transition-transform",
              open && "rotate-180",
            )}
          />
        </button>
      </h1>

      <div id={bodyId} hidden={!open}>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          {map.project.tagline}
        </p>
        <p className="mt-1 font-mono text-[10px] text-muted-foreground/70">
          {map.project.date}
        </p>

        <TopSection
          title="Models"
          icon="ai-agent"
          items={map.topModels}
          fallback="ai-agent"
          className="mt-6"
        />
        <TopSection
          title="Tools"
          icon="hexagon-filled"
          items={map.topTools}
          fallback="hexagon-filled"
          className="mt-6"
        />
        <TopSection
          title="Integrations"
          icon="sitemap-filled"
          items={map.topIntegrations}
          fallback="world-filled"
          className="mt-6"
        />
      </div>
    </div>
  );
}
