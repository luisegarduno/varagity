/**
 * Chat-error presentation: map the wire's `{code, message}` (spec_v2 §4.1's
 * structured envelope — the in-band SSE `error` event and pre-stream 503s
 * share it) to an actionable banner descriptor.
 *
 * The API's dependency preflight emits `<service>_unreachable` codes
 * (`es_unreachable`, `llamacpp_unreachable`, …); the client itself adds
 * `network_error` when the fetch never reached the API. Anything else is
 * unrecognized and surfaces its raw code + message.
 */

/** What the error banner renders for one chat failure. */
export interface ChatErrorDescriptor {
  /** Headline, e.g. `"Elasticsearch is unreachable"`. */
  title: string;
  /** One-line human hint (`null` when the title says it all). */
  hint: string | null;
  /** A shell one-liner worth showing in a code pill (`null` when none). */
  command: string | null;
  /** The banner's affordance: retry the question, or visit the corpus. */
  action: "retry" | "corpus" | null;
  /** Whether to surface the raw `code: message` line (unrecognized errors). */
  raw: boolean;
}

/** Friendly names for the `<service>_unreachable` code prefixes. */
const SERVICE_NAMES: Record<string, string> = {
  es: "Elasticsearch",
  elasticsearch: "Elasticsearch",
  llamacpp: "The chat model",
  llm: "The chat model",
  postgres: "Postgres",
  pg: "Postgres",
  infinity: "The embedding service",
  embeddings: "The embedding service",
  prefect: "Prefect",
};

const COMPOSE_UP = "docker compose up -d --wait";

/**
 * Describe one chat failure for the error banner.
 *
 * Args:
 *   error: The structured `{code, message}` the turn carries (HTTP status,
 *     when there was one, is already folded away upstream).
 *
 * Returns:
 *   The banner descriptor: title, hint, optional command pill and action.
 */
export function describeChatError(error: {
  code: string;
  message: string;
}): ChatErrorDescriptor {
  if (error.code.endsWith("_unreachable")) {
    const service = error.code.slice(0, -"_unreachable".length);
    const name =
      SERVICE_NAMES[service] ??
      service.charAt(0).toUpperCase() + service.slice(1);
    return {
      title: `${name} is unreachable`,
      hint: "Is the stack up?",
      command: COMPOSE_UP,
      action: "retry",
      raw: false,
    };
  }
  if (error.code === "network_error") {
    return {
      title: "Can't reach the Varagity API",
      hint: "Start the stack, then try again:",
      command: COMPOSE_UP,
      action: "retry",
      raw: false,
    };
  }
  return {
    title: "Something went wrong",
    hint: null,
    command: null,
    action: "retry",
    raw: true,
  };
}
