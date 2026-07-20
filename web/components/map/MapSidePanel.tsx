/**
 * The floating side panel over the map canvas: project name + tagline, then
 * the Models / Tools / Integrations callouts with their vendored favicons —
 * foglamp's left card, restyled onto the house tokens.
 */

"use client";

import { useState } from "react";

import type { CodebaseMap, TopItem } from "@/lib/codebase-map";

import { Glyph, iconSrc, modelIconDomain, type GLYPH_PATHS } from "./MapNode";

/** One callout row: favicon (with glyph fallback) + label. */
function TopRow({
  item,
  fallback,
}: {
  item: TopItem;
  fallback: keyof typeof GLYPH_PATHS;
}) {
  const [failed, setFailed] = useState(false);
  const domain = modelIconDomain(item.label, item.domain);
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

/** The panel card, absolutely positioned over the canvas by the view. */
export function MapSidePanel({ map }: { map: CodebaseMap }) {
  return (
    <div className="pointer-events-auto absolute top-4 left-4 z-20 flex max-h-[70%] w-60 flex-col overflow-y-auto rounded-[28px] border border-border bg-card/95 p-5 text-card-foreground backdrop-blur-sm [box-shadow:var(--map-card-shadow)] sm:w-64 dark:border-transparent">
      <h1 className="font-heading text-xl leading-tight font-normal">
        {map.project.name}
        <span className="sr-only"> codebase map</span>
      </h1>
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
  );
}
