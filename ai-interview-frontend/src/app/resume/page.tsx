"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  Sparkles,
  Briefcase,
  GraduationCap,
  Mail,
  Phone,
  Award,
  Building,
} from "lucide-react";

type UploadStatus = "idle" | "uploading" | "success" | "error";

export type ParsedResume = {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
  skills?: string[];
  experience_years?: number | null;
  education?: Array<any>;
  experience?: Array<any>;
  projects?: Array<any>;
  summary?: string | null;
  file_url?: string | null;
  [k: string]: any;
};

type ResumeUploaderProps = {
  onReady?: (parsed: ParsedResume, fileUrl?: string | null) => void;
  onStart?: () => void;
};

export default function ResumeUploader(props: ResumeUploaderProps) {
  const { onReady, onStart } = props;
  const router = useRouter();
  const API = process.env.NEXT_PUBLIC_API_URL ?? "";

  const { token } = useAuth();

  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [progress, setProgress] = useState<number>(0);
  const [parsed, setParsed] = useState<ParsedResume | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const MAX_BYTES = 10 * 1024 * 1024;
  const ACCEPTED = [".pdf", ".docx", ".txt"];

  function formatFileSize(bytes: number) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  }

  function validateFile(f: File | null) {
    if (!f) return "No file selected";
    const lower = f.name.toLowerCase();
    if (!ACCEPTED.some((ext) => lower.endsWith(ext)))
      return "Only PDF, DOCX or TXT files are allowed";
    if (f.size > MAX_BYTES) return "File exceeds 10 MB limit";
    return null;
  }

  const handleAuthCheck = () => {
    if (!token) {
      router.push("/auth/login");
      return false;
    }
    return true;
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (!handleAuthCheck()) return;

    const droppedFile = e.dataTransfer.files?.[0] ?? null;
    const v = validateFile(droppedFile);
    if (v) {
      setErrorMsg(v);
      return;
    }
    setFile(droppedFile);
    setUploadStatus("idle");
    setErrorMsg(null);
  };

  const handleFileSelect: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const selectedFile = e.target.files?.[0] ?? null;
    const v = validateFile(selectedFile);
    if (v) {
      setErrorMsg(v);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    setFile(selectedFile);
    setUploadStatus("idle");
    setErrorMsg(null);
  };

  const handleRemove = () => {
    setFile(null);
    setUploadStatus("idle");
    setParsed(null);
    setProgress(0);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  async function handleUpload() {
    if (!handleAuthCheck()) return;
    if (!file) return;
    const v = validateFile(file);
    if (v) {
      setErrorMsg(v);
      return;
    }

    setUploadStatus("uploading");
    setProgress(0);
    setErrorMsg(null);

    try {
      const form = new FormData();
      form.append("file", file, file.name);

      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const endpoint = (API ? API.replace(/\/$/, "") : "") + "/process-resume";
        xhr.open("POST", endpoint);
        xhr.responseType = "json";

        if (token) {
          xhr.setRequestHeader("Authorization", `Bearer ${token}`);
        }

        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) {
            const pct = Math.round((ev.loaded / ev.total) * 100);
            setProgress(pct);
          }
        };

        xhr.onload = () => {
          const status = xhr.status;
          let body: any = null;
          try {
            body = xhr.response ?? (xhr.responseText ? JSON.parse(xhr.responseText) : {});
          } catch (e) {
            body = { error: "Invalid JSON response from server" };
          }

          if (status >= 200 && status < 300) {
            const parsedBody: ParsedResume = body.parsed ?? body;
            setParsed(parsedBody);
            setUploadStatus("success");

            try {
              const fileUrl = parsedBody.file_url ?? parsedBody.resume_url ?? null;
              onReady?.(parsedBody, fileUrl);
            } catch (cbErr) {
              console.warn("onReady callback threw:", cbErr);
            }
            resolve();
          } else {
            const msg = body?.error ?? JSON.stringify(body ?? { status: status });
            setErrorMsg(msg);
            setUploadStatus("error");
            reject(new Error(msg));
          }
        };

        xhr.onerror = () => {
          setErrorMsg("Network error during upload");
          setUploadStatus("error");
          reject(new Error("Network error during upload"));
        };

        xhr.send(form);
      });
    } catch (err: any) {
      setErrorMsg(err?.message || String(err));
      setUploadStatus("error");
    }
  }

  // --- SAFE RENDER HELPERS ---
  function safeRender(value: any): React.ReactNode {
    if (value === null || value === undefined) return null;
    if (React.isValidElement(value)) return value;
    const t = typeof value;
    if (t === "string" || t === "number" || t === "boolean") return value;
    if (Array.isArray(value)) return value.map((v, i) => <React.Fragment key={i}>{safeRender(v)}</React.Fragment>);
    try {
      // Attempt to pick common display properties for objects to keep JSON small
      if (typeof value === "object") {
        if (value.name && typeof value.name === "string") return value.name;
        if (value.title && typeof value.title === "string") return value.title;
        if (value.company && typeof value.company === "string") return value.company;
        if (value.text && typeof value.text === "string") return value.text;
        // fallback to JSON with limited depth
        return JSON.stringify(value, Object.keys(value).slice(0, 8));
      }
    } catch (e) {
      return String(value);
    }
    return String(value);
  }

  function renderEducation(edu?: Array<any>) {
    if (!edu || !Array.isArray(edu) || edu.length === 0) return null;
    return (
      <div className="space-y-3">
        {edu.map((e, idx) => {
          if (typeof e === "object" && e !== null) {
            const institution = typeof e.institution === "string" ? e.institution : (e.school || "Unknown");
            const degree = typeof e.degree === "string" ? e.degree : (e.program || "");
            const start = typeof e.start_date === "string" ? e.start_date : (e.start || "");
            const end = typeof e.end_date === "string" ? e.end_date : (e.end || "");

            return (
              <div key={idx} className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="font-semibold text-slate-800">{safeRender(institution)}</div>
                    <div className="text-sm text-slate-600 mt-1">{safeRender(degree)}</div>
                  </div>
                  <div className="text-xs text-slate-500 bg-slate-100 px-3 py-1 rounded-full whitespace-nowrap ml-3">
                    {safeRender(start)} {start && end ? "—" : ""} {safeRender(end)}
                  </div>
                </div>
              </div>
            );
          }
          return <div key={idx} className="p-4 bg-white rounded border">{safeRender(e)}</div>;
        })}
      </div>
    );
  }

  function renderExperience(exp?: Array<any>) {
    if (!exp || !Array.isArray(exp) || exp.length === 0) return null;

    return (
      <div className="space-y-3">
        {exp.map((job, idx) => {
          if (typeof job !== "object" || job === null) {
            return (
              <div key={idx} className="p-4 border rounded">
                {safeRender(job)}
              </div>
            );
          }

          const company = typeof job.company === "string" ? job.company : (job.company?.name ?? "Company");
          const position = typeof job.position === "string" ? job.position : (job.title ?? "Role");
          const duration = typeof job.duration === "string" ? job.duration : (job.date_range ?? "");

          let achievements: string[] = [];
          if (Array.isArray(job.achievements)) {
            achievements = job.achievements.map((a: any) => {
              if (typeof a === "string" || typeof a === "number") return String(a);
              if (a && typeof a === "object") {
                // prefer common text fields
                if (typeof a.text === "string") return a.text;
                if (typeof a.description === "string") return a.description;
                if (typeof a.title === "string") return a.title;
                return JSON.stringify(a, Object.keys(a).slice(0, 6));
              }
              return String(a);
            });
          } else if (typeof job.achievements === "string") {
            achievements = [job.achievements];
          } else if (job.achievements && typeof job.achievements === "object") {
            achievements = [safeRender(job.achievements) as string];
          }

          return (
            <div key={idx} className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <div className="font-semibold text-slate-800">{safeRender(company)}</div>
                  <div className="text-sm text-indigo-600 font-medium">{safeRender(position)}</div>
                </div>
                {duration && (
                  <div className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                    {safeRender(duration)}
                  </div>
                )}
              </div>

              {achievements.length > 0 && (
                <ul className="list-disc list-inside text-xs text-slate-600 space-y-1 mt-2">
                  {achievements.map((ach, i) => (
                    <li key={i}>{safeRender(ach)}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  // Debug scan to help locate raw objects that might accidentally be rendered
  useEffect(() => {
    if (!parsed) return;
    function findObjects(o: any, path = "") {
      if (o === null || o === undefined) return;
      if (typeof o === "object" && !Array.isArray(o) && !React.isValidElement(o)) {
        const keys = Object.keys(o).sort().join(",");
        if (keys.includes("company") && keys.includes("position")) {
          console.warn("Found job-like object at", path || "parsed", o);
        }
        Object.entries(o).forEach(([k, v]) => findObjects(v, path ? `${path}.${k}` : k));
      } else if (Array.isArray(o)) {
        o.forEach((v, i) => findObjects(v, `${path}[${i}]`));
      }
    }
    try {
      findObjects(parsed);
    } catch (e) {
      console.warn("scan failed", e);
    }
  }, [parsed]);

  return (
    <div className="min-h-[90vh] bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="inline-flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-xl">
                <Sparkles className="text-white" size={20} />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-slate-800">Resume Parser</h1>
                <p className="text-sm text-slate-500">Upload a resume to extract details</p>
              </div>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-3">
            <button
              type="button"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-white/90 border border-slate-100 shadow-sm hover:shadow-md transition"
              onClick={() => {
                setParsed(null);
                setFile(null);
                setUploadStatus("idle");
                setProgress(0);
                setErrorMsg(null);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            >
              <X size={16} /> Reset
            </button>

            <button
              type="button"
              onClick={() => {
                if (handleAuthCheck()) fileInputRef.current?.click();
              }}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg shadow-lg hover:scale-[1.02] transition-transform ${
                token ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white" : "bg-slate-200 text-slate-600"
              }`}
            >
              <Upload size={16} /> {token ? "Upload" : "Login to upload"}
            </button>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-white/95 rounded-3xl shadow-2xl p-6 border border-slate-100">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            {/* LEFT: Upload Area */}
            <div className="space-y-6 relative">
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => { if (handleAuthCheck()) fileInputRef.current?.click(); }}
                className={`relative rounded-2xl p-8 text-center cursor-pointer transition-all border-2 ${
                  isDragging ? "border-indigo-200 bg-indigo-50 scale-[1.01]" : "border-dashed border-slate-200 hover:border-indigo-300"
                }`}
              >
                {!file ? (
                  <div>
                    <div className="mx-auto w-20 h-20 rounded-full flex items-center justify-center mb-5 bg-slate-100">
                      <Upload className="text-slate-400" size={36} />
                    </div>
                    <h3 className="text-xl font-semibold text-slate-800 mb-2">Drop resume here</h3>
                    <p className="text-slate-500 mb-6">PDF, DOCX, TXT</p>
                  </div>
                ) : (
                  <div className="text-left">
                    <div className="flex items-center gap-3 mb-4">
                      <FileText className="text-indigo-600" size={24} />
                      <div>
                        <div className="font-semibold text-slate-800">{safeRender(file.name)}</div>
                        <div className="text-xs text-slate-500">{formatFileSize(file.size)}</div>
                      </div>
                    </div>

                    {uploadStatus === "uploading" && (
                      <div className="w-full bg-slate-100 rounded-full h-2 mb-4">
                        <div style={{ width: `${progress}%` }} className="h-2 bg-indigo-600 rounded-full" />
                      </div>
                    )}

                    {uploadStatus !== "uploading" && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleUpload(); }}
                        className="w-full py-2 bg-indigo-600 text-white rounded-lg font-medium"
                      >
                        Parse Resume
                      </button>
                    )}
                  </div>
                )}
                <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" onChange={handleFileSelect} className="hidden" />
              </div>
              {errorMsg && <div className="text-sm text-rose-600 bg-rose-50 p-2 rounded">{safeRender(errorMsg)}</div>}
            </div>

            {/* RIGHT: Preview */}
            <div className="space-y-6">
              {!parsed ? (
                <div className="h-full p-12 bg-slate-50 border border-slate-100 rounded-2xl flex flex-col items-center justify-center text-center text-slate-400">
                  <FileText size={48} className="mb-4 opacity-20" />
                  <div>Parsed data will appear here</div>
                </div>
              ) : (
                <div className="space-y-5 animate-in fade-in slide-in-from-right-4 duration-500">
                  {/* Name Card */}
                  <div className="p-5 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 text-white shadow-lg">
                    <h2 className="text-2xl font-bold">{safeRender(parsed.name) || "Candidate"}</h2>
                    <div className="mt-2 space-y-1 text-white/90 text-sm">
                      {parsed.email && <div className="flex items-center gap-2"><Mail size={14} /> {safeRender(parsed.email)}</div>}
                      {parsed.phone && <div className="flex items-center gap-2"><Phone size={14} /> {safeRender(parsed.phone)}</div>}
                    </div>
                    {parsed.experience_years !== null && parsed.experience_years !== undefined && (
                      <div className="mt-4 inline-flex items-center gap-2 px-3 py-1 bg-white/20 rounded-full text-xs font-medium">
                        <Briefcase size={12} /> {safeRender(parsed.experience_years)} Years Exp.
                      </div>
                    )}
                  </div>

                  {/* Summary */}
                  {parsed.summary && (
                    <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-2 mb-2 text-indigo-600 font-semibold">
                        <Award size={18} /> Summary
                      </div>
                      <p className="text-sm text-slate-600 leading-relaxed">{safeRender(parsed.summary)}</p>
                    </div>
                  )}

                  {/* Skills */}
                  {parsed.skills && parsed.skills.length > 0 && (
                    <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                      <div className="flex flex-wrap gap-2">
                        {parsed.skills.map((s, i) => (
                          <span key={i} className="px-2 py-1 bg-indigo-50 text-indigo-700 rounded text-xs font-medium border border-indigo-100">{safeRender(s)}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Experience */}
                  {parsed.experience && parsed.experience.length > 0 && (
                    <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-2 mb-3 text-blue-600 font-semibold">
                        <Building size={18} /> Experience
                      </div>
                      {renderExperience(parsed.experience)}
                    </div>
                  )}

                  {/* Education */}
                  {parsed.education && parsed.education.length > 0 && (
                    <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-2 mb-3 text-emerald-600 font-semibold">
                        <GraduationCap size={18} /> Education
                      </div>
                      {renderEducation(parsed.education)}
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex gap-3 pt-2">
                    <button
                      className="flex-1 py-3 bg-emerald-500 text-white rounded-xl font-bold shadow-lg hover:shadow-emerald-200 transition-all"
                      onClick={() => {
                        if (handleAuthCheck()) {
                          if (onStart) onStart();
                          if (parsed) onReady?.(parsed, parsed.file_url ?? null);
                        }
                      }}
                    >
                      Start Interview
                    </button>
                    <button
                      className="px-6 py-3 bg-white border border-slate-200 text-slate-700 rounded-xl font-medium hover:bg-slate-50 transition-all"
                      onClick={() => {
                        setParsed(null);
                        setFile(null);
                        setUploadStatus("idle");
                      }}
                    >
                      New
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
