import os
import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import jwt
from passlib.context import CryptContext
from app.session_store import _connection_dsn, _ensure_table

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super_secret_key_validex_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

@auth_router.post("/register")
def register(user: UserCreate):
    dsn = _connection_dsn()
    if not dsn:
        raise HTTPException(status_code=500, detail="Database not configured")
    _ensure_table(dsn)
    
    import psycopg
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                # check existing
                cur.execute("SELECT id FROM users WHERE username = %s", (user.username,))
                if cur.fetchone():
                    raise HTTPException(status_code=400, detail="Username already registered")
                
                hashed_pw = get_password_hash(user.password)
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                    (user.username, hashed_pw)
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return {"message": "User created successfully", "user_id": new_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@auth_router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    dsn = _connection_dsn()
    if not dsn:
        raise HTTPException(status_code=500, detail="Database not configured")
    _ensure_table(dsn)
    
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash, is_admin FROM users WHERE username = %s", (form_data.username,))
            row = cur.fetchone()
            if not row or not verify_password(form_data.password, row[1]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            access_token = create_access_token(data={"sub": form_data.username, "user_id": row[0], "is_admin": bool(row[2])})
            return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user_id(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[int]:
    """Returns the user_id if token is valid, else None. It allows anonymous access."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        return user_id
    except jwt.PyJWTError:
        return None

async def get_current_admin_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Returns the user dict if admin, otherwise raises 403 Forbidden."""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("is_admin"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires admin privileges")
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
