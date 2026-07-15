import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import "./styles.css";

const configuredApiBase = import.meta.env.VITE_API_BASE;
const API_BASE = configuredApiBase === "same-origin"
  ? ""
  : configuredApiBase || "http://127.0.0.1:8000";

function cn(...parts) {
  return parts.filter(Boolean).join(" ");
}

function answerStrategyLabel(topK) {
  return ["", "最快", "快速", "均衡", "充分", "深入"][topK] || "均衡";
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message = data?.detail || data?.message || `请求失败: ${response.status}`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return data;
}

function looksLikeInlineMath(value) {
  const text = value.trim();
  if (!text || text.length > 120) return false;
  if (/[\u4e00-\u9fa5]/.test(text)) return false;
  return /\\[a-zA-Z]+|[_^=<>]|[+\-*/]|Δ|∫|∑|∞|lim|frac|sqrt|sin|cos|tan|ln|log|[a-zA-Z]\s*\(|[a-zA-Z]'\s*/.test(text);
}

function normalizeMathDelimiters(content) {
  if (!content) return "";

  let text = content
    .replace(/\\\[/g, "$$")
    .replace(/\\\]/g, "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");

  text = text.replace(/\(([^()\n]{1,120})\)/g, (match, inner) => {
    return looksLikeInlineMath(inner) ? `$${inner.trim()}$` : match;
  });

  return text;
}

function MathMarkdown({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: ({ children }) => <p>{children}</p>,
      }}
    >
      {normalizeMathDelimiters(content)}
    </ReactMarkdown>
  );
}

function App() {
  const [health, setHealth] = useState("checking");
  const [readiness, setReadiness] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedKb, setSelectedKb] = useState("default");
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [activeJob, setActiveJob] = useState(null);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [messages, setMessages] = useState([]);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const activeKb = useMemo(
    () => knowledgeBases.find((item) => item.knowledge_base_id === selectedKb),
    [knowledgeBases, selectedKb],
  );

  async function refreshBasics() {
    try {
      const status = await request("/health");
      setHealth(status.status || "ok");
    } catch {
      setHealth("offline");
    }

    try {
      setReadiness(await request("/api/readiness"));
    } catch {
      setReadiness(null);
    }

    try {
      const result = await request("/api/eval/latest?method=hybrid");
      setMetrics(result.metrics);
    } catch {
      setMetrics(null);
    }

    await refreshKnowledgeBases();
    await refreshJobs();
  }

  async function refreshKnowledgeBases() {
    try {
      const items = await request("/api/knowledge-bases");
      setKnowledgeBases(items);
      if (selectedKb !== "default" && !items.some((item) => item.knowledge_base_id === selectedKb)) {
        setSelectedKb("default");
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshJobs() {
    try {
      const items = await request("/api/jobs?limit=20");
      setJobs(items);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshBasics();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    if (!activeJob?.job_id) return;
    if (pollRef.current) window.clearInterval(pollRef.current);

    pollRef.current = window.setInterval(async () => {
      try {
        const job = await request(`/api/jobs/${activeJob.job_id}`);
        setActiveJob(job);
        if (job.status === "success" || job.status === "failed") {
          window.clearInterval(pollRef.current);
          pollRef.current = null;
          await refreshKnowledgeBases();
          await refreshJobs();
          if (job.status === "success") {
            setSelectedKb(job.knowledge_base_id);
          }
        }
      } catch (err) {
        setError(err.message);
      }
    }, 2500);
  }, [activeJob?.job_id]);

  async function uploadAndBuild(event) {
    event.preventDefault();
    if (!file) return;
    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      const uploaded = await request("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const job = await request("/api/index/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: uploaded.document_id }),
      });
      setActiveJob({ ...job, progress: 0, message: "等待构建", error: "" });
      setFile(null);
      event.currentTarget.reset();
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function deleteKnowledgeBase(knowledgeBaseId) {
    if (!knowledgeBaseId || knowledgeBaseId === "default") return;
    const confirmed = window.confirm("确定删除这个知识库吗？对应索引文件会从本地 storage 中移除。");
    if (!confirmed) return;

    setError("");
    try {
      await request(`/api/knowledge-bases/${knowledgeBaseId}`, {
        method: "DELETE",
      });
      if (selectedKb === knowledgeBaseId) {
        setSelectedKb("default");
      }
      await refreshKnowledgeBases();
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    }
  }

  function jobLabel(job) {
    return job.filename || job.knowledge_base_name || job.document_id;
  }

  function statusIcon(status) {
    if (status === "success" || status === "ready") return "✅";
    if (status === "failed") return "❌";
    if (status === "running" || status === "building") return "⏳";
    return "•";
  }

  function readinessLabel(check) {
    if (!check) return "检查中";
    if (check.status === "ready") return "就绪";
    if (check.status === "download_required") return "需下载";
    return "缺失";
  }

  function readinessTone(check) {
    if (!check) return "warn";
    if (check.status === "ready") return "good";
    if (check.status === "download_required") return "warn";
    return "bad";
  }

  async function ask(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    const userMessage = {
      role: "user",
      question: trimmed,
      knowledge_base_id: selectedKb,
    };
    setMessages((items) => [...items, userMessage]);
    setQuestion("");
    setAsking(true);
    setError("");

    try {
      const result = await request("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          top_k: topK,
          knowledge_base_id: selectedKb,
        }),
      });
      setMessages((items) => [...items, { role: "assistant", ...result }]);
    } catch (err) {
      setError(err.message);
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          answer: err.message || "请求失败，请检查后端服务、模型或 API Key 配置。",
          query: trimmed,
          contexts: [],
          confidence: {},
          knowledge_base_id: selectedKb,
        },
      ]);
    } finally {
      setAsking(false);
    }
  }

  async function sendFeedback(message, rating) {
    const comment = message.feedbackComment || "";
    try {
      await request("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: message.query || "",
          answer: message.answer || "",
          rating,
          comment,
          contexts: message.contexts || [],
          top_rerank_score: message.confidence?.top_rerank_score ?? null,
          knowledge_base_id: message.knowledge_base_id || selectedKb,
        }),
      });
      setMessages((items) =>
        items.map((item) => (item === message ? { ...item, feedback: rating } : item)),
      );
    } catch (err) {
      setError(err.message);
    }
  }

  function updateFeedbackComment(message, value) {
    setMessages((items) =>
      items.map((item) => (item === message ? { ...item, feedbackComment: value } : item)),
    );
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark">∑</div>
          <div>
            <h1>MathRAG</h1>
            <p>智能数学知识检索</p>
          </div>
        </div>

        <section className="panel">
          <div className="panelHead">
            <h2>📊 运行状态</h2>
            <button type="button" className="iconButton" onClick={refreshBasics} title="刷新">
              ↻
            </button>
          </div>
          <div className="statusRows">
            <div>
              <span>🖥️ 后端</span>
              <strong className={cn("pill", health === "ok" ? "good" : "bad")}>{health}</strong>
            </div>
            <div>
              <span>默认索引</span>
              <strong
                className={cn("pill", readinessTone(readiness?.checks?.default_index))}
                title={readiness?.checks?.default_index?.detail}
              >
                {readinessLabel(readiness?.checks?.default_index)}
              </strong>
            </div>
            <div>
              <span>Embedding</span>
              <strong
                className={cn("pill", readinessTone(readiness?.checks?.embedding_model))}
                title={readiness?.checks?.embedding_model?.detail}
              >
                {readinessLabel(readiness?.checks?.embedding_model)}
              </strong>
            </div>
            <div>
              <span>Reranker</span>
              <strong
                className={cn("pill", readinessTone(readiness?.checks?.reranker_model))}
                title={readiness?.checks?.reranker_model?.detail}
              >
                {readinessLabel(readiness?.checks?.reranker_model)}
              </strong>
            </div>
            <div>
              <span>DeepSeek Key</span>
              <strong
                className={cn("pill", readinessTone(readiness?.checks?.deepseek_api_key))}
                title={readiness?.checks?.deepseek_api_key?.detail}
              >
                {readinessLabel(readiness?.checks?.deepseek_api_key)}
              </strong>
            </div>
            <div>
              <span>📈 Recall@5</span>
              <strong>{metrics ? `${Math.round(metrics.recall_at_5 * 100)}%` : "-"}</strong>
            </div>
            <div>
              <span>📝 评测题数</span>
              <strong>{metrics?.question_count ?? "-"}</strong>
            </div>
          </div>
          {!!readiness?.blockers?.length && (
            <div className="readinessIssues">
              {readiness.blockers.map((blocker) => <p key={blocker}>{blocker}</p>)}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panelHead">
            <h2>📚 知识库</h2>
            <button type="button" className="iconButton" onClick={refreshKnowledgeBases} title="刷新知识库">
              ↻
            </button>
          </div>
          <select value={selectedKb} onChange={(event) => setSelectedKb(event.target.value)}>
            <option value="default">📖 默认知识库</option>
            {knowledgeBases.map((item) => (
              <option key={item.knowledge_base_id} value={item.knowledge_base_id}>
                {item.name} {item.status === 'ready' ? '✅' : '⏳'}
              </option>
            ))}
          </select>
          {activeKb && <p className="muted" style={{ marginTop: '4px' }}>状态：{activeKb.status}</p>}
          {selectedKb !== "default" && (
            <button
              type="button"
              className="dangerButton"
              onClick={() => deleteKnowledgeBase(selectedKb)}
            >
              删除当前知识库
            </button>
          )}
        </section>

        <form className="panel upload" onSubmit={uploadAndBuild}>
          <h2>📄 上传教材</h2>
          <label className="filePicker">
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <div className="file-label">
              <span className="icon">📚</span>
              <span>{file ? file.name : "点击选择 PDF 文件"}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-light)' }}>
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : '支持 .pdf 格式'}
              </span>
            </div>
          </label>
          <button type="submit" className="primary" disabled={!file || uploading}>
            {uploading ? "⏳ 上传中..." : "🚀 上传并构建"}
          </button>
          {activeJob && (
            <div className="job">
              <div>
                <strong>{activeJob.status === 'success' ? '✅' : activeJob.status === 'failed' ? '❌' : '⏳'} {activeJob.status}</strong>
                <span>{activeJob.progress ?? 0}%</span>
              </div>
              <progress value={activeJob.progress ?? 0} max="100" />
              <p>{activeJob.error || activeJob.message || activeJob.knowledge_base_id}</p>
            </div>
          )}
        </form>

        <section className="panel">
          <div className="panelHead">
            <h2>🧾 任务历史</h2>
            <button type="button" className="iconButton" onClick={refreshJobs} title="刷新任务">
              ↻
            </button>
          </div>
          <div className="jobHistory">
            {jobs.length === 0 && <p className="muted">暂无构建任务</p>}
            {jobs.map((job) => (
              <div className="jobItem" key={job.job_id}>
                <div className="jobItemTop">
                  <strong title={jobLabel(job)}>
                    {statusIcon(job.status)} {jobLabel(job)}
                  </strong>
                  <span>{job.progress ?? 0}%</span>
                </div>
                <progress value={job.progress ?? 0} max="100" />
                <p>{job.error || job.message || job.status}</p>
                <small>{job.knowledge_base_id}</small>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>💬 智能问答</h2>
            <p>📖 {selectedKb === "default" ? "默认知识库" : selectedKb}</p>
          </div>
          <label className="answerStrategy" htmlFor="answer-strategy">
            <span className="strategyHeader">
              <span>回答策略</span>
              <strong>{answerStrategyLabel(topK)}</strong>
            </span>
            <input
              id="answer-strategy"
              className="strategyRange"
              type="range"
              min="1"
              max="5"
              step="1"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              aria-label="回答策略"
              aria-valuetext={answerStrategyLabel(topK)}
            />
            <span className="strategyLabels" aria-hidden="true">
              <span>更快</span>
              <span>参考更充分</span>
            </span>
          </label>
        </header>

        {error && <div className="error">{error}</div>}

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <span style={{ fontSize: '48px', marginBottom: '8px' }}>🧮</span>
              <h3>开始探索数学知识</h3>
              <p>输入你的高数问题，AI 将基于知识库为你解答</p>
              <p style={{ fontSize: '13px', color: 'var(--text-light)', marginTop: '4px' }}>
                💡 例如：洛必达法则的适用条件是什么？
              </p>
            </div>
          )}

          {messages.map((message, index) =>
            message.role === "user" ? (
              <article className="message user" key={`${message.role}-${index}`}>
                <p>{message.question}</p>
              </article>
            ) : (
              <article className="message assistant" key={`${message.role}-${index}`}>
                <div className="answer">
                  <MathMarkdown content={message.answer} />
                </div>
                <div className="metaLine">
                  <span>
                    📊 置信度：
                    {message.confidence?.top_rerank_score == null
                      ? "-"
                      : message.confidence.top_rerank_score.toFixed(3)}
                  </span>
                  <span>📄 引用：{message.contexts?.length || 0}</span>
                </div>
                {!!message.contexts?.length && (
                  <details>
                    <summary>📖 查看检索片段</summary>
                    <div className="contexts">
                      {message.contexts.slice(0, 3).map((context, contextIndex) => (
                        <div className="context" key={context.vector_id || contextIndex}>
                          <strong>
                            📄 {context.source_file || context.file || context.title || `片段 ${contextIndex + 1}`}
                          </strong>
                          <p>{context.content}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                <div className="feedback">
                  <button
                    type="button"
                    className={cn(message.feedback === "up" && "selected")}
                    onClick={() => sendFeedback(message, "up")}
                  >
                    👍 有用
                  </button>
                  <button
                    type="button"
                    className={cn(message.feedback === "down" && "selected")}
                    onClick={() => sendFeedback(message, "down")}
                  >
                    👎 不准
                  </button>
                </div>
                <form
                  className="feedbackComment"
                  onSubmit={(event) => {
                    event.preventDefault();
                    sendFeedback(message, message.feedback || "down");
                  }}
                >
                  <textarea
                    value={message.feedbackComment || ""}
                    onChange={(event) => updateFeedbackComment(message, event.target.value)}
                    placeholder="写下具体反馈，比如哪里不准、缺了什么、希望怎么回答..."
                    rows="2"
                  />
                  <button
                    type="submit"
                    disabled={!message.feedbackComment?.trim()}
                  >
                    提交文字反馈
                  </button>
                </form>
              </article>
            ),
          )}
          {asking && <div className="thinking">🧠 正在检索知识并生成答案...</div>}
        </div>

        <form className="composer" onSubmit={ask}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="✏️ 输入你的数学问题..."
            rows="3"
          />
          <button type="submit" className="primary" disabled={asking || !question.trim()}>
            {asking ? '⏳ 思考中...' : '🚀 发送'}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
