import { describe, expect, it } from "vitest";

import { describeChatError } from "@/lib/errors";

describe("describeChatError", () => {
  it("names the down service for *_unreachable codes", () => {
    const descriptor = describeChatError({
      code: "es_unreachable",
      message: "elasticsearch unreachable — is the stack up?",
    });
    expect(descriptor.title).toBe("Elasticsearch is unreachable");
    expect(descriptor.command).toBe("docker compose up -d --wait");
    expect(descriptor.action).toBe("retry");
    expect(descriptor.raw).toBe(false);
  });

  it("knows every preflight service prefix", () => {
    expect(
      describeChatError({ code: "llamacpp_unreachable", message: "" }).title,
    ).toBe("The chat model is unreachable");
    expect(
      describeChatError({ code: "postgres_unreachable", message: "" }).title,
    ).toBe("Postgres is unreachable");
    expect(
      describeChatError({ code: "infinity_unreachable", message: "" }).title,
    ).toBe("The embedding service is unreachable");
    expect(
      describeChatError({ code: "prefect_unreachable", message: "" }).title,
    ).toBe("Prefect is unreachable");
  });

  it("falls back to the capitalized prefix for unknown services", () => {
    expect(
      describeChatError({ code: "redis_unreachable", message: "" }).title,
    ).toBe("Redis is unreachable");
  });

  it("treats network_error as the API itself being down", () => {
    const descriptor = describeChatError({
      code: "network_error",
      message: "TypeError: fetch failed",
    });
    expect(descriptor.title).toBe("Can't reach the Varagity API");
    expect(descriptor.command).toBe("docker compose up -d --wait");
    expect(descriptor.action).toBe("retry");
    expect(descriptor.raw).toBe(false);
  });

  it("surfaces unrecognized codes raw, still offering a retry", () => {
    const descriptor = describeChatError({
      code: "pipeline_error",
      message: "ValueError: boom",
    });
    expect(descriptor.title).toBe("Something went wrong");
    expect(descriptor.hint).toBeNull();
    expect(descriptor.command).toBeNull();
    expect(descriptor.action).toBe("retry");
    expect(descriptor.raw).toBe(true);
  });
});
