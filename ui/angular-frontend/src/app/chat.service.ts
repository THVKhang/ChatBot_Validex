import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ChatApiResponse } from './chat.models';
import { DeleteReportResponse } from './chat.models';
import { HealthResponse } from './chat.models';
import { MetricsResponse } from './chat.models';
import { ReportDetailResponse } from './chat.models';
import { ReportsListResponse } from './chat.models';
import { SaveReportRequest } from './chat.models';
import { SaveReportResponse } from './chat.models';
import { KnowledgeHealthResponse } from './chat.models';
import { SourceAnalyticsResponse } from './chat.models';
import { ReportStatus } from './chat.models';
import { ChatSessionSummary, ChatSessionHistory } from './chat.models';

export interface StreamEvent {
  type: 'meta' | 'done' | 'error' | 'thinking';
  data: any;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiBase = 'http://localhost:8000/api';

  constructor(private readonly http: HttpClient) {}

  sendMessage(prompt: string, sessionId: string | null): Observable<ChatApiResponse> {
    return this.http.post<ChatApiResponse>(`${this.apiBase}/chat`, {
      prompt,
      session_id: sessionId,
    });
  }

  /**
   * Stream chat via SSE — returns a callback-based approach using fetch + ReadableStream.
   */
  async sendMessageStream(
    prompt: string,
    sessionId: string | null,
    onEvent: (event: StreamEvent) => void,
  ): Promise<void> {
    const response = await fetch(`${this.apiBase}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, session_id: sessionId }),
    });

    if (!response.ok || !response.body) {
      onEvent({ type: 'error', data: { error: `HTTP ${response.status}` } });
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent({ type: currentEvent as StreamEvent['type'], data });
          } catch {
            // ignore parse errors
          }
          currentEvent = '';
        }
      }
    }
  }

  healthCheck(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiBase}/health`);
  }

  getMetrics(): Observable<MetricsResponse> {
    return this.http.get<MetricsResponse>(`${this.apiBase}/metrics`);
  }

  getSourceAnalytics(): Observable<SourceAnalyticsResponse> {
    return this.http.get<SourceAnalyticsResponse>(`${this.apiBase}/source-analytics`);
  }

  getKnowledgeHealth(): Observable<KnowledgeHealthResponse> {
    return this.http.get<KnowledgeHealthResponse>(`${this.apiBase}/knowledge/health`);
  }

  saveReport(request: SaveReportRequest): Observable<SaveReportResponse> {
    return this.http.post<SaveReportResponse>(`${this.apiBase}/reports`, request);
  }

  listReports(limit = 20): Observable<ReportsListResponse> {
    return this.http.get<ReportsListResponse>(`${this.apiBase}/reports`, {
      params: { limit },
    });
  }

  getReport(reportId: string): Observable<ReportDetailResponse> {
    return this.http.get<ReportDetailResponse>(`${this.apiBase}/reports/${reportId}`);
  }

  updateReportStatus(reportId: string, status: ReportStatus): Observable<ReportDetailResponse> {
    return this.http.patch<ReportDetailResponse>(`${this.apiBase}/reports/${reportId}/status`, {
      status,
    });
  }

  deleteReport(reportId: string): Observable<DeleteReportResponse> {
    return this.http.delete<DeleteReportResponse>(`${this.apiBase}/reports/${reportId}`);
  }

  getChatSessions(limit = 50): Observable<ChatSessionSummary[]> {
    return this.http.get<ChatSessionSummary[]>(`${this.apiBase}/chat/sessions`, {
      params: { limit },
    });
  }

  getChatSession(sessionId: string): Observable<ChatSessionHistory> {
    return this.http.get<ChatSessionHistory>(`${this.apiBase}/chat/sessions/${sessionId}`);
  }

  exportChat(markdown: string, format: 'docx' | 'html' = 'docx'): Observable<Blob> {
    return this.http.post(
      `${this.apiBase}/chat/export`,
      { markdown, format },
      { responseType: 'blob' }
    );
  }

  getAdminMetrics(): Observable<any> {
    return this.http.get<any>(`${this.apiBase}/admin/metrics`);
  }

  getAdminUsers(): Observable<any> {
    return this.http.get<any>(`${this.apiBase}/admin/users`);
  }

  triggerIngest(): Observable<any> {
    return this.http.post(
      `${this.apiBase}/admin/ingest`,
      {}
    );
  }

  getIngestStatus(): Observable<any> {
    return this.http.get(
      `${this.apiBase}/admin/ingest/status`
    );
  }

  uploadFile(file: File): Observable<{filename: string, extracted_text: string}> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<{filename: string, extracted_text: string}>(`${this.apiBase}/chat/upload`, formData);
  }

  getCrawlHistory(limit = 20): Observable<any> {
    return this.http.get<any>(`${this.apiBase}/admin/crawl-history`, { params: { limit } });
  }

  triggerDiscovery(): Observable<any> {
    return this.http.post<any>(`${this.apiBase}/admin/discover`, {});
  }
}

