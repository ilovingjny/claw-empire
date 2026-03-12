import fs from "node:fs";
import path from "node:path";
import type { ChildProcess } from "node:child_process";
import { decryptSecret } from "../../../../oauth/helpers.ts";
import type { ApiProviderRow } from "./types.ts";

const DEFAULT_MODEL_OVERRIDES: Record<string, string> = {
  "alibaba-coding-plan-openai": "qwen3-coder-plus",
  "alibaba-coding-plan-anthropic": "qwen3-coder-plus",
};

type DbLike = {
  prepare: (sql: string) => {
    get: (...args: any[]) => unknown;
  };
};

type CreateApiProviderToolsDeps = {
  db: DbLike;
  logsDir: string;
  activeProcesses: Map<string, ChildProcess>;
  broadcast: (event: string, payload: unknown) => void;
  normalizeStreamChunk: (raw: Buffer | string, opts?: { dropCliNoise?: boolean }) => string;
  handleTaskRunComplete: (taskId: string, exitCode: number) => void;
  createSafeLogStreamOps: (logStream: any) => {
    safeWrite: (text: string) => boolean;
    safeEnd: (onDone?: () => void) => void;
    isClosed: () => boolean;
  };
  parseSSEStream: (
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal,
    safeWrite: (text: string) => boolean,
    taskId?: string,
  ) => Promise<void>;
  parseGeminiSSEStream: (
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal,
    safeWrite: (text: string) => boolean,
    taskId?: string,
  ) => Promise<void>;
};

export function createApiProviderTools(deps: CreateApiProviderToolsDeps) {
  const {
    db,
    logsDir,
    activeProcesses,
    broadcast,
    normalizeStreamChunk,
    handleTaskRunComplete,
    createSafeLogStreamOps,
    parseSSEStream,
    parseGeminiSSEStream,
  } = deps;

  async function parseAnthropicSSEStream(
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal,
    safeWrite: (text: string) => boolean,
    taskId?: string,
  ): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";

    const processLine = (trimmed: string) => {
      if (!trimmed || trimmed.startsWith(":")) return;
      if (!trimmed.startsWith("data: ")) return;
      if (trimmed === "data: [DONE]") return;
      try {
        const data = JSON.parse(trimmed.slice(6));
        if (data.type === "content_block_delta" && data.delta?.text) {
          const text = normalizeStreamChunk(data.delta.text);
          if (!text) return;
          safeWrite(text);
          if (taskId) {
            broadcast("cli_output", { task_id: taskId, stream: "stdout", data: text });
          }
        }
      } catch {
        /* ignore */
      }
    };

    for await (const chunk of body as AsyncIterable<Uint8Array>) {
      if (signal.aborted) break;
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) processLine(line.trim());
    }
    if (buffer.trim()) processLine(buffer.trim());
  }

  function getApiProviderById(providerId: string): ApiProviderRow | null {
    return (
      (db.prepare("SELECT * FROM api_providers WHERE id = ?").get(providerId) as unknown as ApiProviderRow) ?? null
    );
  }

  function parseModelsCache(value: string | null): string[] {
    if (!value) return [];
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map((item) => String(item ?? "").trim()).filter(Boolean) : [];
    } catch {
      return [];
    }
  }

  function looksLikeHtmlResponse(value: string): boolean {
    return /^\s*<!doctype html/i.test(value) || /^\s*<html\b/i.test(value);
  }

  function summarizeUpstreamErrorBody(value: string): string {
    const trimmed = value.trim();
    if (!trimmed) return "";
    if (looksLikeHtmlResponse(trimmed)) {
      return "Upstream returned HTML instead of a streaming API response. Check that the Base URL and API key target the direct API endpoint.";
    }
    try {
      const parsed = JSON.parse(trimmed) as {
        error?: { message?: unknown } | string;
        message?: unknown;
      };
      if (typeof parsed.error === "string" && parsed.error.trim()) return parsed.error.trim();
      if (
        parsed.error &&
        typeof parsed.error === "object" &&
        "message" in parsed.error &&
        typeof parsed.error.message === "string" &&
        parsed.error.message.trim()
      ) {
        return parsed.error.message.trim();
      }
      if (typeof parsed.message === "string" && parsed.message.trim()) return parsed.message.trim();
    } catch {
      // fall through to raw text summary
    }
    return trimmed.slice(0, 500);
  }

  function resolveDefaultModel(provider: ApiProviderRow, cachedModels: readonly string[]): string | null {
    const presetKey = typeof provider.preset_key === "string" ? provider.preset_key : "";
    const preferredModel = presetKey ? DEFAULT_MODEL_OVERRIDES[presetKey] : "";
    if (preferredModel) {
      return cachedModels.includes(preferredModel) ? preferredModel : preferredModel;
    }
    return cachedModels[0] ?? null;
  }

  function resolveApiProviderModel(provider: ApiProviderRow, requestedModel: string | null): string {
    if (requestedModel) return requestedModel;
    const cachedModels = parseModelsCache(provider.models_cache);
    const defaultModel = resolveDefaultModel(provider, cachedModels);
    if (defaultModel) {
      return defaultModel;
    }
    throw new Error(
      `No model specified for API provider '${provider.name}'. ` +
        `Please select a model in the agent settings or run a connection test first to cache available models.`,
    );
  }

  function normalizeApiBaseUrl(rawUrl: string): string {
    let url = rawUrl.replace(/\/+$/, "");
    url = url.replace(/\/(v\d+)\/(chat\/completions|models|messages)$/i, "/$1");
    url = url.replace(/\/v1beta\/models\/.+$/i, "/v1beta");
    return url;
  }

  function buildApiProviderRequest(
    provider: ApiProviderRow,
    model: string,
    prompt: string,
    projectPath: string,
  ): { url: string; headers: Record<string, string>; body: string } {
    const apiKey = provider.api_key_enc ? decryptSecret(provider.api_key_enc) : "";
    const baseUrl = normalizeApiBaseUrl(provider.base_url);

    if (provider.type === "anthropic") {
      const messagesUrl = baseUrl.endsWith("/v1") ? `${baseUrl}/messages` : `${baseUrl}/v1/messages`;
      return {
        url: messagesUrl,
        headers: {
          "x-api-key": apiKey,
          "Content-Type": "application/json",
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model,
          max_tokens: 16384,
          stream: true,
          messages: [{ role: "user", content: prompt }],
          system: `You are a coding assistant. Project path: ${projectPath}`,
        }),
      };
    }

    if (provider.type === "google") {
      const googleBase = baseUrl.endsWith("/v1beta") ? baseUrl : `${baseUrl}/v1beta`;
      const url = `${googleBase}/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;
      return {
        url,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          systemInstruction: { parts: [{ text: `You are a coding assistant. Project path: ${projectPath}` }] },
        }),
      };
    }

    const chatUrl = /\/v\d+$/.test(baseUrl) ? `${baseUrl}/chat/completions` : `${baseUrl}/v1/chat/completions`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    if (provider.type === "openrouter") {
      headers["HTTP-Referer"] = "https://claw-empire.app";
      headers["X-Title"] = "Claw-Empire";
    }

    return {
      url: chatUrl,
      headers,
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: `You are a coding assistant. Project path: ${projectPath}` },
          { role: "user", content: prompt },
        ],
        stream: true,
      }),
    };
  }

  async function executeApiProviderAgent(
    prompt: string,
    projectPath: string,
    logStream: fs.WriteStream,
    signal: AbortSignal,
    taskId?: string,
    apiProviderId?: string | null,
    apiModel?: string | null,
    safeWriteOverride?: (text: string) => boolean,
  ): Promise<void> {
    const safeWrite = safeWriteOverride ?? createSafeLogStreamOps(logStream).safeWrite;

    if (!apiProviderId) {
      throw new Error("No API provider configured for this agent. Set api_provider_id first.");
    }

    const provider = getApiProviderById(apiProviderId);
    if (!provider) {
      throw new Error(`API provider not found: ${apiProviderId}`);
    }
    if (!provider.enabled) {
      throw new Error(`API provider '${provider.name}' is disabled.`);
    }

    const model = resolveApiProviderModel(provider, apiModel ?? null);
    const header = `[api:${provider.type}] Provider: ${provider.name}, Model: ${model}\n---\n`;
    safeWrite(header);
    if (taskId) broadcast("cli_output", { task_id: taskId, stream: "stderr", data: header });

    const req = buildApiProviderRequest(provider, model, prompt, projectPath);

    const resp = await fetch(req.url, {
      method: "POST",
      headers: req.headers,
      body: req.body,
      signal,
    });

    if (!resp.ok) {
      const text = await resp.text();
      const detail = summarizeUpstreamErrorBody(text) || `upstream returned ${resp.status}`;
      throw new Error(`API provider '${provider.name}' error (${resp.status}): ${detail}`);
    }

    const contentType = resp.headers.get("content-type") ?? "";
    if (contentType && !/text\/event-stream/i.test(contentType)) {
      const text = await resp.text().catch(() => "");
      const detail =
        summarizeUpstreamErrorBody(text) ||
        `Unexpected content-type '${contentType}'. The provider did not return a streaming API response.`;
      throw new Error(`API provider '${provider.name}' returned unexpected response: ${detail}`);
    }

    if (!resp.body) {
      throw new Error(`API provider '${provider.name}' returned an empty response body.`);
    }

    if (provider.type === "anthropic") {
      await parseAnthropicSSEStream(resp.body, signal, safeWrite, taskId);
    } else if (provider.type === "google") {
      await parseGeminiSSEStream(resp.body, signal, safeWrite, taskId);
    } else {
      await parseSSEStream(resp.body, signal, safeWrite, taskId);
    }

    safeWrite(`\n---\n[api:${provider.type}] Done.\n`);
    if (taskId) {
      broadcast("cli_output", { task_id: taskId, stream: "stderr", data: `\n---\n[api:${provider.type}] Done.\n` });
    }
  }

  function launchApiProviderAgent(
    taskId: string,
    apiProviderId: string | null,
    apiModel: string | null,
    prompt: string,
    projectPath: string,
    logPath: string,
    controller: AbortController,
    fakePid: number,
  ): void {
    const logStream = fs.createWriteStream(logPath, { flags: "a" });
    const { safeWrite, safeEnd } = createSafeLogStreamOps(logStream);
    safeWrite(`\n===== task run start ${new Date().toISOString()} | provider=api =====\n`);

    const promptPath = path.join(logsDir, `${taskId}.prompt.txt`);
    fs.writeFileSync(promptPath, prompt, "utf8");

    const mockProc = {
      pid: fakePid,
      kill: () => {
        controller.abort();
        return true;
      },
    } as unknown as ChildProcess;
    activeProcesses.set(taskId, mockProc);

    const runTask = (async () => {
      let exitCode = 0;
      try {
        await executeApiProviderAgent(
          prompt,
          projectPath,
          logStream,
          controller.signal,
          taskId,
          apiProviderId,
          apiModel,
          safeWrite,
        );
      } catch (err: any) {
        exitCode = 1;
        if (err.name !== "AbortError") {
          const msg = normalizeStreamChunk(`[api] Error: ${err.message}\n`);
          safeWrite(msg);
          broadcast("cli_output", { task_id: taskId, stream: "stderr", data: msg });
          console.error(`[Claw-Empire] API provider agent error (task ${taskId}): ${err.message}`);
        } else {
          const msg = normalizeStreamChunk(`[api] Aborted by user\n`);
          safeWrite(msg);
          broadcast("cli_output", { task_id: taskId, stream: "stderr", data: msg });
        }
      } finally {
        await new Promise<void>((resolve) => safeEnd(resolve));
        try {
          fs.unlinkSync(promptPath);
        } catch {
          /* ignore */
        }
        handleTaskRunComplete(taskId, exitCode);
      }
    })();

    runTask.catch(() => {});
  }

  return {
    executeApiProviderAgent,
    launchApiProviderAgent,
  };
}
