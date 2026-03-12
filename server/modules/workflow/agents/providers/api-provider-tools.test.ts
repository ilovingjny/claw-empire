import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiProviderTools } from "./api-provider-tools.ts";
import type { ApiProviderRow } from "./types.ts";

function createHarness(provider: ApiProviderRow) {
  const parseSSEStream = vi.fn().mockResolvedValue(undefined);
  const parseGeminiSSEStream = vi.fn().mockResolvedValue(undefined);

  const tools = createApiProviderTools({
    db: {
      prepare: () => ({
        get: () => provider,
      }),
    },
    logsDir: "C:\\temp",
    activeProcesses: new Map(),
    broadcast: vi.fn(),
    normalizeStreamChunk: (raw) => String(raw),
    handleTaskRunComplete: vi.fn(),
    createSafeLogStreamOps: () => ({
      safeWrite: () => true,
      safeEnd: (onDone?: () => void) => onDone?.(),
      isClosed: () => false,
    }),
    parseSSEStream,
    parseGeminiSSEStream,
  });

  return { ...tools, parseSSEStream, parseGeminiSSEStream };
}

describe("api provider tools", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("prefers the supported Bailian default model when no explicit model is set", async () => {
    const provider: ApiProviderRow = {
      id: "provider-1",
      name: "Bailian Coding Plan",
      type: "openai",
      base_url: "https://coding-intl.dashscope.aliyuncs.com/v1",
      api_key_enc: null,
      preset_key: "alibaba-coding-plan-openai",
      enabled: 1,
      models_cache: JSON.stringify(["qwen3.5-plus", "qwen3-coder-plus", "qwen3-max-2026-01-23"]),
      models_cached_at: 1_717_171_717_000,
    };
    const { executeApiProviderAgent, parseSSEStream } = createHarness(provider);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response("data: [DONE]\n\n", { status: 200, headers: { "content-type": "text/event-stream" } }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await executeApiProviderAgent(
      "hello",
      "C:\\repo",
      {} as never,
      new AbortController().signal,
      undefined,
      provider.id,
      null,
      () => true,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      model: "qwen3-coder-plus",
      stream: true,
    });
    expect(parseSSEStream).toHaveBeenCalledTimes(1);
  });

  it("summarizes unexpected HTML responses instead of leaking raw markup", async () => {
    const provider: ApiProviderRow = {
      id: "provider-2",
      name: "OpenCode Go",
      type: "openai",
      base_url: "https://opencode.ai/zen/go/v1",
      api_key_enc: null,
      preset_key: "opencode-go-openai",
      enabled: 1,
      models_cache: JSON.stringify(["glm-5", "kimi-k2.5"]),
      models_cached_at: 1_717_171_717_000,
    };
    const { executeApiProviderAgent, parseSSEStream } = createHarness(provider);
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("<!DOCTYPE html><html><head></head><body>web page</body></html>", {
        status: 200,
        headers: { "content-type": "text/html; charset=utf-8" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      executeApiProviderAgent(
        "hello",
        "C:\\repo",
        {} as never,
        new AbortController().signal,
        undefined,
        provider.id,
        null,
        () => true,
      ),
    ).rejects.toThrow("Upstream returned HTML instead of a streaming API response");

    expect(parseSSEStream).not.toHaveBeenCalled();
  });

  it("includes the target URL when fetch fails before a response arrives", async () => {
    const provider: ApiProviderRow = {
      id: "provider-3",
      name: "OpenCode Go",
      type: "openai",
      base_url: "https://opencode.ai/zen/go/v1",
      api_key_enc: null,
      preset_key: "opencode-go-openai",
      enabled: 1,
      models_cache: JSON.stringify(["glm-5"]),
      models_cached_at: 1_717_171_717_000,
    };
    const { executeApiProviderAgent, parseSSEStream } = createHarness(provider);
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    await expect(
      executeApiProviderAgent(
        "hello",
        "C:\\repo",
        {} as never,
        new AbortController().signal,
        undefined,
        provider.id,
        null,
        () => true,
      ),
    ).rejects.toThrow("Failed to reach https://opencode.ai/zen/go/v1/chat/completions");

    expect(parseSSEStream).not.toHaveBeenCalled();
  });
});
