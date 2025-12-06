// src/components/ResumeUploader.tsx
"use client";

import React, { useState, useRef } from "react";
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
} from "lucide-react";

type UploadStatus = "idle" | "uploading" | "success" | "error";

type ParsedResume = {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
  skills?: string[];
  experience_years?: number | null;
  education?: Array<Record<string, any>>;
  summary?: string | null;
  [k: string]: any;
};

export default function ResumeUploader() {
  const API = process.env.NEXT_PUBLIC_API_URL ?? "";

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
    if (!ACCEPTED.some((ext) => lower.endsWith(ext))) return "Only PDF, DOCX or TXT files are allowed";
    if (f.size > MAX_BYTES) return "File exceeds 10 MB limit";
    return null;
  }

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
            const parsedBody = body.parsed ?? body;
            setParsed(parsedBody);
            setUploadStatus("success");
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

  function renderEducation(edu?: Array<Record<string, any>>) {
    if (!edu || edu.length === 0) return <div className="text-sm text-slate-500">No education details found</div>;
    return (
      <div className="space-y-3">
        {edu.map((e, idx) => {
          const institution = typeof e === "string" ? e : e.institution || e.school || "";
          const degree = typeof e === "string" ? "" : e.degree || e.program || "";
          const field = typeof e === "string" ? "" : e.field || e.major || "";
          const start = typeof e === "string" ? "" : e.start_date || e.start || "";
          const end = typeof e === "string" ? "" : e.end_date || e.end || "";
          return (
            <div key={idx} className="p-4 bg-white rounded-xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="font-semibold text-slate-800">{institution || degree || "Education"}</div>
                  <div className="text-sm text-slate-600 mt-1">{degree}{field ? ` • ${field}` : ""}</div>
                </div>
                <div className="text-xs text-slate-500 bg-slate-100 px-3 py-1 rounded-full whitespace-nowrap ml-3">
                  {start || ""}{start && end ? " — " : ""}{end || ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

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
                <p className="text-sm text-slate-500">Upload a resume to extract contact info, skills, education and more</p>
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
              title="Reset"
            >
              <X size={16} />
              Reset
            </button>

            <button
              type="button"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg hover:scale-[1.02] transition-transform"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={16} />
              Upload
            </button>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-white/95 rounded-3xl shadow-2xl p-6 border border-slate-100">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            {/* LEFT: Upload Area */}
            <div className="space-y-6 relative">
              {/* Drag overlay */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                role="button"
                aria-label="Upload resume"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
                }}
                className={`relative rounded-2xl p-8 text-center cursor-pointer transition-all border-2 ${
                  isDragging
                    ? "border-indigo-200 bg-indigo-50 scale-[1.01] shadow-md"
                    : "border-dashed border-slate-200 hover:border-indigo-300 hover:bg-slate-50"
                }`}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className={`absolute inset-0 rounded-2xl pointer-events-none transition-opacity ${isDragging ? "opacity-100" : "opacity-0"}`}>
                  <div className="w-full h-full rounded-2xl border-2 border-indigo-300 border-dashed" />
                </div>

                {!file ? (
                  <div>
                    <div className="mx-auto w-20 h-20 rounded-full flex items-center justify-center mb-5 transition-all bg-slate-100">
                      <Upload className={`transition-colors ${isDragging ? "text-indigo-600" : "text-slate-400"}`} size={36} />
                    </div>

                    <h3 className="text-xl font-semibold text-slate-800 mb-2">{isDragging ? "Drop it here" : "Drop your resume here"}</h3>
                    <p className="text-slate-500 mb-6">PDF, DOCX, TXT • Max 10 MB</p>

                    <div className="flex items-center justify-center gap-3">
                      <div
                        className="inline-flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg font-medium shadow hover:scale-105 transition-transform"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <Upload size={18} />
                        Browse file
                      </div>

                      <div className="text-sm text-slate-400">or paste a URL / drag & drop</div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4 text-left">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 bg-indigo-50 p-3 rounded-lg">
                        <FileText className="text-indigo-600" size={26} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <h3 className="font-semibold text-slate-800 text-lg truncate">{file.name}</h3>
                            <p className="text-sm text-slate-500">{formatFileSize(file.size)} • {file.name.split(".").pop()?.toUpperCase()}</p>
                          </div>

                          <div className="flex items-center gap-2">
                            {uploadStatus === "success" && <div className="text-emerald-600 flex items-center gap-1"><CheckCircle size={16} /> Parsed</div>}
                            <button
                              type="button"
                              onClick={handleRemove}
                              className="p-2 rounded-md text-slate-500 hover:text-rose-500 hover:bg-rose-50 transition"
                              aria-label="Remove file"
                            >
                              <X size={18} />
                            </button>
                          </div>
                        </div>

                        {uploadStatus === "error" && errorMsg && (
                          <div className="mt-3 text-sm text-rose-600 bg-rose-50 px-3 py-2 rounded">{errorMsg}</div>
                        )}

                        {uploadStatus === "uploading" && (
                          <div className="mt-3">
                            <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                              <div>Uploading…</div>
                              <div>{progress}%</div>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-2">
                              <div style={{ width: `${progress}%` }} className="h-2 bg-gradient-to-r from-indigo-500 to-purple-600 transition-all" />
                            </div>
                          </div>
                        )}

                        {uploadStatus !== "uploading" && (
                          <div className="mt-4 flex gap-3">
                            <button
                              onClick={handleUpload}
                              disabled={uploadStatus === "uploading"}
                              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow hover:scale-[1.02] transition-transform"
                              type="button"
                            >
                              <Sparkles size={16} />
                              Parse resume
                            </button>

                            <button
                              onClick={() => fileInputRef.current?.click()}
                              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 text-slate-700 bg-white hover:bg-slate-50 transition"
                              type="button"
                            >
                              Replace
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>

              {/* Small helper / tips */}
              <div className="text-xs text-slate-500">
                Tip: clean headings and consistent formatting usually yields better parsing results.
              </div>
            </div>

            {/* RIGHT: Preview */}
            <div className="space-y-6">
              {!parsed ? (
                // placeholder / skeleton while nothing parsed yet
                <div className="h-full p-6 rounded-2xl bg-slate-50 border border-slate-100 shadow-sm flex flex-col items-center justify-center text-center">
                  {uploadStatus === "uploading" ? (
                    <div className="w-48 animate-pulse">
                      <div className="h-6 bg-slate-200 rounded mb-3" />
                      <div className="h-4 bg-slate-200 rounded mb-2" />
                      <div className="h-4 bg-slate-200 rounded mb-2" />
                      <div className="h-4 bg-slate-200 rounded" />
                    </div>
                  ) : (
                    <div className="text-center max-w-xs">
                      <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow">
                        <FileText className="text-slate-400" size={26} />
                      </div>
                      <h3 className="font-semibold text-slate-700 mb-2">Waiting for resume...</h3>
                      <p className="text-sm text-slate-500">Upload a file and press “Parse resume” to extract details</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Name Card */}
                  <div className="p-4 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 text-white shadow-lg">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="text-sm opacity-90 mb-1">Candidate</div>
                        <h2 className="text-2xl font-bold mb-2">{parsed.name || "Unknown"}</h2>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-white/90">
                            <Mail size={15} />
                            <span className="text-sm">{parsed.email || "Not provided"}</span>
                          </div>
                          <div className="flex items-center gap-2 text-white/90">
                            <Phone size={15} />
                            <span className="text-sm">{parsed.phone || "Not provided"}</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-white/20 backdrop-blur-sm px-4 py-3 rounded-xl text-center">
                        <Briefcase size={22} className="mx-auto mb-1" />
                        <div className="text-xl font-bold">{parsed.experience_years ?? "—"}</div>
                        <div className="text-xs opacity-90">years</div>
                      </div>
                    </div>
                  </div>

                  {/* Summary */}
                  {parsed.summary && (
                    <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <Award className="text-indigo-500" size={18} />
                        <h3 className="font-semibold text-slate-800">Summary</h3>
                      </div>
                      <p className="text-slate-700 leading-relaxed">{parsed.summary}</p>
                    </div>
                  )}

                  {/* Skills */}
                  <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                    <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                      <Sparkles className="text-purple-500" size={18} />
                      Skills
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {(parsed.skills && parsed.skills.length > 0) ? (
                        parsed.skills.map((s: string, i: number) => (
                          <span
                            key={i}
                            className="px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-sm font-medium border border-indigo-100"
                          >
                            {s}
                          </span>
                        ))
                      ) : (
                        <div className="text-sm text-slate-500">No skills found</div>
                      )}
                    </div>
                  </div>

                  {/* Education */}
                  <div className="p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                    <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
                      <GraduationCap className="text-indigo-500" size={18} />
                      Education
                    </h3>
                    {renderEducation(parsed.education)}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <button className="flex-1 py-3 rounded-lg font-semibold bg-emerald-500 text-white hover:shadow-lg transition-all" type="button">
                      Start Interview
                    </button>
                    <button
                      onClick={() => {
                        setParsed(null);
                        setFile(null);
                        setUploadStatus("idle");
                        setProgress(0);
                        if (fileInputRef.current) fileInputRef.current.value = "";
                      }}
                      className="px-5 py-3 rounded-lg font-semibold border border-slate-200 text-slate-700 hover:bg-slate-50 transition-all"
                      type="button"
                    >
                      Upload New
                    </button>
                  </div>

                  {/* Raw JSON */}
                  <details className="text-xs">
                    <summary className="cursor-pointer text-slate-500 hover:text-slate-700 font-medium">View raw JSON</summary>
                    <pre className="mt-3 p-3 bg-slate-900 text-green-400 rounded-lg overflow-auto text-xs">
                      {JSON.stringify(parsed, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer small */}
        <div className="mt-4 text-xs text-slate-400 text-center">
          Processing is done locally or by your API (ensure CORS for cross-origin endpoints).
        </div>
      </div>
    </div>
  );
}
