const { useState, useEffect, useCallback, useRef } = React;

const API = "/api";

/* ──────────────────────────── Tiny Components ──────────────────────────── */

function Badge({ type, children }) {
  return <span className={`badge badge-${type}`}>{children}</span>;
}

function SkillTypeBadge({ type }) {
  const t = (type || "").toLowerCase();
  return <Badge type={t}>{t}</Badge>;
}

function StatusBadge({ skill }) {
  if (!skill.is_active) return <Badge type="inactive">INACTIVE</Badge>;
  if (skill.is_expired) return <Badge type="expired">EXPIRED</Badge>;
  return <Badge type="active">ACTIVE</Badge>;
}

function TagsList({ tags }) {
  if (!tags || tags.length === 0) return null;
  return <span>{tags.map(t => <span className="tag" key={t}>{t}</span>)}</span>;
}

function SimpleMarkdown({ text }) {
  if (!text) return null;
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^```(\w*)\n([\s\S]*?)```/gm, '<pre><code>$2</code></pre>')
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  return <div className="md-content" dangerouslySetInnerHTML={{ __html: html }} />;
}

function RelevanceScoreBar({ score }) {
  const pct = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? "#22c55e" : pct >= 50 ? "#eab308" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 60, height: 6, background: "#334155", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width .3s" }} />
      </div>
      <span style={{ fontSize: 11, color: "#94a3b8", minWidth: 32 }}>{score.toFixed(1)}%</span>
    </div>
  );
}

/* ──────────────────────────── Shared Modal Shell ───────────────────────── */

function ModalShell({ onClose, title, children, wide }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} onClick={e => e.stopPropagation()}>
        <span className="close" onClick={onClose}>&times;</span>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

/* ──────────────────────────── Create Skill Modal ───────────────────────── */

function CreateSkillModal({ onClose, onCreated, repos }) {
  const [form, setForm] = useState({
    name: "", description: "", body: "", skill_type: "IMPLEMENTATION",
    repo_name: "global", tags: "", ttl_days: ""
  });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setError("");
    if (!form.name || !form.description || !form.body) {
      setError("Name, description, body are required");
      return;
    }
    setSaving(true);
    try {
      const res = await fetch(`${API}/skills`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed");
      }
      onCreated();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <ModalShell onClose={onClose} title="Create New Skill" wide>
      {error && <p className="error-text">{error}</p>}
      <div className="form-grid">
        <label>
          Name (kebab-case)*
          <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value.replace(/[^a-z0-9-]/g, '').toLowerCase() })} placeholder="cors-fix-nextjs-api" />
        </label>
        <label>
          Type
          <select value={form.skill_type} onChange={e => setForm({ ...form, skill_type: e.target.value })}>
            {["IMPLEMENTATION", "WORKFLOW", "TROUBLESHOOTING", "ARCHITECTURE", "RULE"].map(t =>
              <option key={t} value={t}>{t}</option>
            )}
          </select>
        </label>
        <label>
          Repo
          <select value={form.repo_name} onChange={e => setForm({ ...form, repo_name: e.target.value })}>
            <option value="global">global</option>
            {(repos || []).map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
          </select>
        </label>
        <label>
          Tags (comma-separated)
          <input value={form.tags} onChange={e => setForm({ ...form, tags: e.target.value })} placeholder="cors, nextjs, api" />
        </label>
        <label>
          TTL (days, 0=never)
          <input type="number" value={form.ttl_days} onChange={e => setForm({ ...form, ttl_days: e.target.value })} placeholder="0" />
        </label>
      </div>
      <label className="full-width">
        Description*
        <input value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} placeholder="Fix CORS errors on Next.js API routes" />
      </label>
      <label className="full-width">
        Body (Markdown)*
        <textarea rows={10} value={form.body} onChange={e => setForm({ ...form, body: e.target.value })}
          placeholder={"# Title\n\n## When to Use\n- ...\n\n## Solution\n```typescript\n```\n\n## Lessons Learned\n- **V1**: ..."} />
      </label>
      <div className="form-actions">
        <button onClick={onClose}>Cancel</button>
        <button className="btn-primary" onClick={save} disabled={saving}>{saving ? "Saving..." : "Create Skill"}</button>
      </div>
    </ModalShell>
  );
}

/* ──────────────────────────── Edit Skill Modal ─────────────────────────── */

function EditSkillModal({ skill, onClose, onSaved, repos }) {
  const [form, setForm] = useState({
    description: skill.description || "",
    body: "",
    skill_type: skill.skill_type || "IMPLEMENTATION",
    repo_name: skill.repo_name || "global",
    tags: (skill.tags || []).join(", "),
    ttl_days: skill.ttl_days != null ? String(skill.ttl_days) : "",
  });
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [loadingBody, setLoadingBody] = useState(true);

  // Fetch current body content
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/skills/${skill.id}/content`);
        const data = await res.json();
        setForm(f => ({ ...f, body: data.raw || data.body || "" }));
      } catch (e) {
        console.error("Failed to load skill body", e);
      } finally {
        setLoadingBody(false);
      }
    })();
  }, [skill.id]);

  const save = async () => {
    setError("");
    if (!form.description || !form.body) {
      setError("Description and body are required");
      return;
    }
    setSaving(true);
    try {
      const payload = {
        description: form.description,
        body: form.body,
        skill_type: form.skill_type,
        repo_name: form.repo_name,
        tags: form.tags,
        ttl_days: form.ttl_days ? Number(form.ttl_days) : 0,
      };
      const res = await fetch(`${API}/skills/${encodeURIComponent(skill.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to update skill");
      }
      onSaved();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <ModalShell onClose={onClose} title={`Edit Skill: ${skill.display_name || skill.id}`} wide>
      {error && <p className="error-text">{error}</p>}
      <p style={{ fontSize: 13, color: "#64748b", marginBottom: 12 }}>
        Editing <code>{skill.id}</code> — V{skill.version_number}
      </p>
      {loadingBody ? (
        <p style={{ color: "#94a3b8" }}>Loading current content...</p>
      ) : (
        <>
          <div className="form-grid">
            <label>
              Type
              <select value={form.skill_type} onChange={e => setForm({ ...form, skill_type: e.target.value })}>
                {["IMPLEMENTATION", "WORKFLOW", "TROUBLESHOOTING", "ARCHITECTURE", "RULE"].map(t =>
                  <option key={t} value={t}>{t}</option>
                )}
              </select>
            </label>
            <label>
              Repo
              <select value={form.repo_name} onChange={e => setForm({ ...form, repo_name: e.target.value })}>
                <option value="global">global</option>
                {(repos || []).map(r => <option key={r.name} value={r.name}>{r.name}</option>)}
              </select>
            </label>
            <label>
              Tags (comma-separated)
              <input value={form.tags} onChange={e => setForm({ ...form, tags: e.target.value })} placeholder="cors, nextjs, api" />
            </label>
            <label>
              TTL (days, 0=never)
              <input type="number" value={form.ttl_days} onChange={e => setForm({ ...form, ttl_days: e.target.value })} placeholder="0" />
            </label>
          </div>
          <label className="full-width">
            Description*
            <input value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} placeholder="Skill description" />
          </label>
          <label className="full-width">
            Body (Markdown)*
            <textarea rows={12} value={form.body} onChange={e => setForm({ ...form, body: e.target.value })} />
          </label>
          <div className="form-actions">
            <button onClick={onClose}>Cancel</button>
            <button className="btn-primary" onClick={save} disabled={saving}>{saving ? "Saving..." : "Save Changes"}</button>
          </div>
        </>
      )}
    </ModalShell>
  );
}

/* ──────────────────────── References Display Component ─────────────────── */

function ReferencesDisplay({ skillName }) {
  const [refs, setRefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/skills/${encodeURIComponent(skillName)}/references`);
        if (!res.ok) throw new Error("Failed to load references");
        setRefs(await res.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [skillName]);

  if (loading) return <div className="references-section"><p style={{ color: "#94a3b8" }}>Loading references...</p></div>;
  if (error) return <div className="references-section"><p className="error-text">{error}</p></div>;

  const outgoing = refs?.outgoing_references || refs?.referenced_by_this_skill || refs?.references || [];
  const incoming = refs?.incoming_references || refs?.referencing_this_skill || refs?.referenced_by || [];

  return (
    <div className="references-section">
      {/* Outgoing — skills this skill references */}
      <div className="ref-group">
        <h4>
          <span className="ref-icon">&#8594;</span> Referenced Skills
          <span className="ref-count">{Array.isArray(outgoing) ? outgoing.length : 0}</span>
        </h4>
        {Array.isArray(outgoing) && outgoing.length > 0 ? (
          <ul className="ref-list">
            {outgoing.map((r, i) => (
              <li key={i} className="ref-item">
                <code>{typeof r === "string" ? r : r.name || r.skill_name || r.id}</code>
                {r.type && <SkillTypeBadge type={r.type} />}
                {r.description && <span style={{ color: "#64748b", marginLeft: 8, fontSize: 12 }}>"{r.description}"</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="ref-empty">This skill does not reference other skills.</p>
        )}
      </div>

      {/* Incoming — skills that reference this skill */}
      <div className="ref-group">
        <h4>
          <span className="ref-icon">&#8592;</span> Referenced By
          <span className="ref-count">{Array.isArray(incoming) ? incoming.length : 0}</span>
        </h4>
        {Array.isArray(incoming) && incoming.length > 0 ? (
          <ul className="ref-list">
            {incoming.map((r, i) => (
              <li key={i} className="ref-item">
                <code>{typeof r === "string" ? r : r.name || r.skill_name || r.id}</code>
                {r.type && <SkillTypeBadge type={r.type} />}
                {r.description && <span style={{ color: "#64748b", marginLeft: 8, fontSize: 12 }}>"{r.description}"</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="ref-empty">No other skills reference this skill.</p>
        )}
      </div>
    </div>
  );
}

/* ──────────────────────────────── Main App ─────────────────────────────── */

function App() {
  const [stats, setStats] = useState(null);
  const [skills, setSkills] = useState([]);
  const [repos, setRepos] = useState([]);
  const [filters, setFilters] = useState({ repo: "", type: "", search: "", activeOnly: true, showExpired: false });

  // Hybrid search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchActive, setSearchActive] = useState(false);
  const searchTimerRef = useRef(null);

  const [sortField, setSortField] = useState("last_modified_at");
  const [sortDir, setSortDir] = useState("desc");

  const [selectedSkill, setSelectedSkill] = useState(null);
  const [skillContent, setSkillContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editingSkill, setEditingSkill] = useState(null);

  // Export state
  const [exporting, setExporting] = useState(false);

  /* ────── Data fetching ────── */

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, r] = await Promise.all([
        fetch(`${API}/stats`).then(r => r.json()),
        fetch(`${API}/repos`).then(r => r.json()),
      ]);
      setStats(s);
      setRepos(r);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  const fetchSkills = useCallback(async () => {
    const params = new URLSearchParams();
    if (filters.repo) params.set("repo", filters.repo);
    if (filters.type) params.set("skill_type", filters.type);
    if (filters.search) params.set("search", filters.search);
    params.set("active_only", filters.activeOnly);
    params.set("expired", filters.showExpired);
    const res = await fetch(`${API}/skills?${params}`);
    setSkills(await res.json());
  }, [filters]);

  useEffect(() => { fetchData(); }, [fetchData]);
  useEffect(() => { fetchSkills(); }, [fetchSkills]);

  /* ────── Hybrid search (POST /api/search with debounce) ────── */

  const doHybridSearch = useCallback(async (query) => {
    if (!query || query.trim().length < 2) {
      setSearchResults(null);
      setSearchActive(false);
      return;
    }
    setSearching(true);
    try {
      const res = await fetch(`${API}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), top_k: 20 }),
      });
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      setSearchResults(Array.isArray(data) ? data : data.results || data.skills || []);
      setSearchActive(true);
    } catch (e) {
      console.error("Hybrid search error:", e);
      setSearchResults([]);
      setSearchActive(true);
    } finally {
      setSearching(false);
    }
  }, []);

  const handleSearchInput = useCallback((value) => {
    setSearchQuery(value);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    if (!value || value.trim().length < 2) {
      setSearchResults(null);
      setSearchActive(false);
      return;
    }
    searchTimerRef.current = setTimeout(() => doHybridSearch(value), 350);
  }, [doHybridSearch]);

  const clearSearch = useCallback(() => {
    setSearchQuery("");
    setSearchResults(null);
    setSearchActive(false);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
  }, []);

  /* ────── Export (GET /api/export) ────── */

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await fetch(`${API}/export`);
      if (!res.ok) throw new Error("Export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `skills-export-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export error:", e);
      alert("Export failed: " + e.message);
    } finally {
      setExporting(false);
    }
  };

  /* ────── Skill actions ────── */

  const openSkill = async (name) => {
    try {
      const [detail, content] = await Promise.all([
        fetch(`${API}/skills/${encodeURIComponent(name)}`).then(r => r.json()),
        fetch(`${API}/skills/${encodeURIComponent(name)}/content`).then(r => r.json()),
      ]);
      setSelectedSkill(detail);
      setSkillContent(content);
    } catch (e) { console.error(e); }
  };

  const deprecateSkill = async (name) => {
    if (!confirm(`Deprecate skill "${name}"?`)) return;
    await fetch(`${API}/skills/${encodeURIComponent(name)}/deprecate`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: "Manual deprecation" }),
    });
    fetchSkills();
    fetchData();
    setSelectedSkill(null);
    setSkillContent(null);
  };

  const deleteSkill = async (name) => {
    if (!confirm(`DELETE skill "${name}" permanently? This cannot be undone.`)) return;
    await fetch(`${API}/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
    fetchSkills();
    fetchData();
    setSelectedSkill(null);
    setSkillContent(null);
  };

  const closeModal = () => {
    setSelectedSkill(null);
    setSkillContent(null);
  };

  /* ────── Sorting ────── */

  const sortedSkills = [...(searchActive && searchResults ? searchResults : skills)].sort((a, b) => {
    let va, vb;
    switch (sortField) {
      case "name": va = a.id || a.name; vb = b.id || b.name; break;
      case "type": va = a.skill_type; vb = b.skill_type; break;
      case "repo": va = a.repo_name; vb = b.repo_name; break;
      case "version": va = a.version_number || 0; vb = b.version_number || 0; break;
      case "status": va = a.is_active ? 2 : a.is_expired ? 0 : 1; vb = b.is_active ? 2 : b.is_expired ? 0 : 1; break;
      case "use_count": va = a.use_count || 0; vb = b.use_count || 0; break;
      default: va = a.last_modified_at || ""; vb = b.last_modified_at || "";
    }
    if (typeof va === "string") {
      return sortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return sortDir === "asc" ? va - vb : vb - va;
  });

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const SortArrow = ({ field }) => {
    if (sortField !== field) return <span className="sort-arrow">&#8693;</span>;
    return <span className="sort-arrow">{sortDir === "asc" ? "&#8593;" : "&#8595;"}</span>;
  };

  if (loading) return <div className="empty"><p>Loading Skills Lab...</p></div>;

  /* ══════════════════════════════ RENDER ══════════════════════════════ */
  return (
    <div>
      {/* ═══════ Header ═══════ */}
      <header>
        <h1>Skills Lab Dashboard</h1>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Badge type="active">SKILL.md v2.1</Badge>
          <button className="btn-secondary" onClick={handleExport} disabled={exporting} title="Export all skills as JSON">
            {exporting ? "⏳ Exporting..." : "⬇ Export JSON"}
          </button>
          <button className="btn-primary" onClick={() => setShowCreate(true)}>+ New Skill</button>
        </div>
      </header>

      {/* ═══════ Stats Cards ═══════ */}
      {stats && (
        <div className="stats">
          <div className="stat-card">
            <div className="number">{stats.total}</div>
            <div className="label">Total Skills</div>
          </div>
          <div className="stat-card">
            <div className="number">{stats.active}</div>
            <div className="label">Active</div>
          </div>
          <div className="stat-card">
            <div className="number">{stats.inactive}</div>
            <div className="label">Inactive</div>
          </div>
          <div className="stat-card">
            <div className="number">{stats.expired}</div>
            <div className="label">Expired</div>
          </div>
          <div className="stat-card">
            <div className="number">{stats.repos?.length || repos.length}</div>
            <div className="label">Repos</div>
          </div>
        </div>
      )}

      {/* ═══════ Hybrid Search Bar ═══════ */}
      <div className="search-bar-container">
        <div className="search-bar">
          <span className="search-icon">&#128269;</span>
          <input
            type="text"
            className="search-input"
            placeholder="Hybrid search (BM25 + Semantic) — type at least 2 characters..."
            value={searchQuery}
            onChange={e => handleSearchInput(e.target.value)}
          />
          {searchQuery && (
            <button className="search-clear" onClick={clearSearch} title="Clear search">&#10005;</button>
          )}
          {searching && <span className="search-spinner">&#8987;</span>}
        </div>
        {searchActive && (
          <div className="search-status">
            {searchResults && searchResults.length > 0
              ? <span className="search-result-count">&#128270; {searchResults.length} result{searchResults.length !== 1 ? "s" : ""} found</span>
              : <span className="search-result-count">No results found</span>
            }
            <button className="search-clear-btn" onClick={clearSearch}>Clear search</button>
          </div>
        )}
      </div>

      {/* ═══════ Filter Bar ═══════ */}
      <div className="filters">
        <input placeholder="Filter skills..." value={filters.search}
          onChange={e => setFilters(f => ({ ...f, search: e.target.value }))} style={{ width: 200 }} />
        <select value={filters.repo} onChange={e => setFilters(f => ({ ...f, repo: e.target.value }))}>
          <option value="">All Repos</option>
          {repos.map(r => <option key={r.name} value={r.name}>{r.name} ({r.skill_count})</option>)}
        </select>
        <select value={filters.type} onChange={e => setFilters(f => ({ ...f, type: e.target.value }))}>
          <option value="">All Types</option>
          <option value="IMPLEMENTATION">Implementation</option>
          <option value="WORKFLOW">Workflow</option>
          <option value="TROUBLESHOOTING">Troubleshooting</option>
          <option value="ARCHITECTURE">Architecture</option>
          <option value="RULE">Rule</option>
        </select>
        <label style={{ fontSize: 13, color: "#94a3b8", cursor: "pointer" }}>
          <input type="checkbox" checked={filters.activeOnly} onChange={e => setFilters(f => ({ ...f, activeOnly: e.target.checked }))} />
          Active only
        </label>
        <label style={{ fontSize: 13, color: "#94a3b8", cursor: "pointer" }}>
          <input type="checkbox" checked={filters.showExpired} onChange={e => setFilters(f => ({ ...f, showExpired: e.target.checked }))} />
          Show expired
        </label>
      </div>

      {/* ═══════ Skills Table ═══════ */}
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort("name")}>Name <SortArrow field="name" /></th>
              <th className="sortable" onClick={() => toggleSort("type")}>Type <SortArrow field="type" /></th>
              <th className="sortable" onClick={() => toggleSort("repo")}>Repo <SortArrow field="repo" /></th>
              <th className="sortable" onClick={() => toggleSort("version")}>V <SortArrow field="version" /></th>
              <th>Tags</th>
              <th className="sortable" onClick={() => toggleSort("status")}>Status <SortArrow field="status" /></th>
              <th className="sortable" onClick={() => toggleSort("use_count")}>Used <SortArrow field="use_count" /></th>
              <th className="sortable" onClick={() => toggleSort("last_modified_at")}>Modified <SortArrow field="last_modified_at" /></th>
              {searchActive && <th>Score</th>}
              <th style={{ width: 110 }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedSkills.length === 0 && (
              <tr>
                <td colSpan={searchActive ? 10 : 9} className="empty">
                  {searchActive ? "No matching skills found" : "No skills found"}
                </td>
              </tr>
            )}
            {sortedSkills.map(s => {
              const score = s.score ?? s.relevance_score ?? s.hybrid_score ?? null;
              return (
                <tr key={s.id || s.name}>
                  <td className="skill-name" onClick={() => openSkill(s.id || s.name)}>
                    {s.id || s.name}
                    {s.display_name && s.display_name !== (s.id || s.name) && (
                      <span style={{ color: "#64748b", marginLeft: 8 }}>{s.display_name}</span>
                    )}
                  </td>
                  <td><SkillTypeBadge type={s.skill_type} /></td>
                  <td>{s.repo_name}</td>
                  <td>V{s.version_number}</td>
                  <td><TagsList tags={s.tags} /></td>
                  <td><StatusBadge skill={s} /></td>
                  <td>{s.use_count ?? 0}</td>
                  <td style={{ fontSize: 12 }}>{s.last_modified_at ? new Date(s.last_modified_at).toLocaleDateString() : "—"}</td>
                  {searchActive && (
                    <td>
                      {score != null ? <RelevanceScoreBar score={score * 100} /> : <span style={{ color: "#475569", fontSize: 11 }}>—</span>}
                    </td>
                  )}
                  <td>
                    <div style={{ display: "flex", gap: 4, justifyContent: "flex-end" }}>
                      <button className="btn-action btn-edit" onClick={(e) => { e.stopPropagation(); setEditingSkill(s); }} title="Edit">
                        &#9998;
                      </button>
                      {s.is_active && (
                        <button className="danger" onClick={(e) => { e.stopPropagation(); deprecateSkill(s.id || s.name); }} title="Deprecate" style={{ padding: "2px 8px" }}>
                          &#10005;
                        </button>
                      )}
                      <button className="danger" onClick={(e) => { e.stopPropagation(); deleteSkill(s.id || s.name); }} title="Delete" style={{ padding: "2px 8px", borderColor: "#991b1b" }}>
                        &#128465;
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ═══════ Skill Detail Modal ═══════ */}
      {selectedSkill && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal modal-wide" onClick={e => e.stopPropagation()}>
            <span className="close" onClick={closeModal}>&times;</span>
            <h2>
              {selectedSkill.display_name || selectedSkill.id}
              <StatusBadge skill={selectedSkill} />
              <SkillTypeBadge type={selectedSkill.skill_type} />
            </h2>
            <p className="meta-line">
              <code>{selectedSkill.id}</code> | V{selectedSkill.version_number} | {selectedSkill.repo_name}
              {" | "}<TagsList tags={selectedSkill.tags} />
              {selectedSkill.use_count != null && <span> | Used {selectedSkill.use_count}x</span>}
            </p>
            <p className="description">{selectedSkill.description}</p>

            {/* Version History */}
            {selectedSkill.lineage_chain && selectedSkill.lineage_chain.length > 0 && (
              <div className="version-history">
                <h3>Version History</h3>
                {selectedSkill.lineage_chain.map((item, i) => (
                  <div key={i} className="version-entry">
                    <span className="version-badge">{item.trigger} V{item.from_version} &rarr; V{item.to_version}</span>
                    <span className="version-reason">{item.reason}</span>
                    {item.source_skill_id && <span className="version-source">from: {item.source_skill_id}</span>}
                  </div>
                ))}
              </div>
            )}

            {/* References Section */}
            <div className="references-section-wrapper">
              <h3>References</h3>
              <ReferencesDisplay skillName={selectedSkill.id} />
            </div>

            {/* SKILL.md Content */}
            {skillContent && (
              <div>
                <h3>SKILL.md Content</h3>
                <div className="skill-md-body">
                  <SimpleMarkdown text={skillContent.raw || skillContent.body} />
                </div>
              </div>
            )}

            {/* Detail modal actions */}
            <div className="form-actions" style={{ marginTop: 16 }}>
              <button className="btn-secondary" onClick={() => { setEditingSkill(selectedSkill); }}>
                &#9998; Edit Skill
              </button>
              {selectedSkill.is_active && (
                <button className="danger" onClick={() => deprecateSkill(selectedSkill.id)}>
                  Deprecate
                </button>
              )}
              <button className="danger" onClick={() => deleteSkill(selectedSkill.id)} style={{ borderColor: "#991b1b" }}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════ Create Skill Modal ═══════ */}
      {showCreate && (
        <CreateSkillModal
          repos={repos}
          onClose={() => setShowCreate(false)}
          onCreated={() => { fetchSkills(); fetchData(); }}
        />
      )}

      {/* ═══════ Edit Skill Modal ═══════ */}
      {editingSkill && (
        <EditSkillModal
          skill={editingSkill}
          repos={repos}
          onClose={() => setEditingSkill(null)}
          onSaved={() => { fetchSkills(); fetchData(); if (selectedSkill?.id === editingSkill.id) closeModal(); }}
        />
      )}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById("root"));
