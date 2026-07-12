"use client";

import { CheckIcon, CopyIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import ReactMarkdown, {
  type Components,
  type ExtraProps,
} from "react-markdown";
import ShikiHighlighter, {
  isInlineCode,
  type Element as ShikiElement,
} from "react-shiki/web";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { Button } from "@/components/ui/button";

/** Debounce a fast-changing value (the ~80 ms anti-flash for streaming). */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Copy code"
      className="absolute top-1.5 right-1.5 opacity-0 transition-opacity group-hover/code:opacity-100 focus-visible:opacity-100"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
    </Button>
  );
}

function CodeBlock({
  className,
  children,
  node,
  ...props
}: React.ComponentProps<"code"> & ExtraProps) {
  // react-shiki bundles its own copy of the hast Element type — runtime
  // shape is identical, so reconcile the duplicate declaration with a cast.
  const inline = node ? isInlineCode(node as unknown as ShikiElement) : true;
  if (inline) {
    return (
      <code
        className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]"
        {...props}
      >
        {children}
      </code>
    );
  }
  const language = /language-(\w+)/.exec(className ?? "")?.[1];
  const code = String(children).replace(/\n$/, "");
  return (
    <span className="group/code relative block">
      <CopyButton text={code} />
      <ShikiHighlighter
        language={language ?? "text"}
        theme={{ light: "github-light", dark: "github-dark" }}
        defaultColor="light"
        showLanguage={false}
        className="text-sm [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:p-3"
      >
        {code}
      </ShikiHighlighter>
    </span>
  );
}

/**
 * Answer markdown: GFM (tables, strikethrough, task lists), fenced code
 * with copy + shiki highlighting, and KaTeX math. Memoized — during
 * streaming, pass the text through `useDebouncedValue` so partial markdown
 * (half-fenced code, unclosed emphasis) doesn't flash re-styles per token.
 * `components` (memoize it — this component is `memo`ed on prop identity)
 * lets callers override elements, e.g. the citation-chip `a` renderer.
 */
export const Markdown = memo(function Markdown({
  text,
  components,
}: {
  text: string;
  components?: Components;
}) {
  return (
    <div className="space-y-3 leading-relaxed [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_table]:block [&_table]:overflow-x-auto [&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1 [&_ul]:list-disc [&_ul]:pl-5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{ code: CodeBlock, ...components }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
