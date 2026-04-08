import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ChatApiResponse } from './chat.models';

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

  healthCheck(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.apiBase}/health`);
  }
}
