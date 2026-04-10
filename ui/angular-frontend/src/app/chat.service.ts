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

  healthCheck(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.apiBase}/health`);
  }

  getMetrics(): Observable<MetricsResponse> {
    return this.http.get<MetricsResponse>(`${this.apiBase}/metrics`);
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

  deleteReport(reportId: string): Observable<DeleteReportResponse> {
    return this.http.delete<DeleteReportResponse>(`${this.apiBase}/reports/${reportId}`);
  }
}
