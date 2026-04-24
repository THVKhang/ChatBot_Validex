import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';

export interface AuthResponse {
  access_token: string;
  token_type: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private baseUrl = 'http://127.0.0.1:8000/api/auth';
  private tokenKey = 'validex_auth_token';
  private usernameKey = 'validex_username';
  
  private authStatus = new BehaviorSubject<boolean>(this.hasToken());
  private currentUser = new BehaviorSubject<string | null>(localStorage.getItem(this.usernameKey));
  private isAdmin = new BehaviorSubject<boolean>(this.checkIfAdmin());

  constructor(private http: HttpClient) {}

  get isLoggedIn$(): Observable<boolean> {
    return this.authStatus.asObservable();
  }

  get currentUser$(): Observable<string | null> {
    return this.currentUser.asObservable();
  }
  
  get isAdmin$(): Observable<boolean> {
    return this.isAdmin.asObservable();
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  hasToken(): boolean {
    return !!this.getToken();
  }
  
  private checkIfAdmin(): boolean {
    const token = this.getToken();
    if (!token) return false;
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return !!payload.is_admin;
    } catch (e) {
      return false;
    }
  }

  login(username: string, password: string): Observable<AuthResponse> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    return this.http.post<AuthResponse>(`${this.baseUrl}/login`, formData).pipe(
      tap(res => {
        localStorage.setItem(this.tokenKey, res.access_token);
        localStorage.setItem(this.usernameKey, username);
        this.authStatus.next(true);
        this.currentUser.next(username);
        this.isAdmin.next(this.checkIfAdmin());
      })
    );
  }

  register(username: string, password: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/register`, { username, password });
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.usernameKey);
    this.authStatus.next(false);
    this.currentUser.next(null);
    this.isAdmin.next(false);
  }
}
