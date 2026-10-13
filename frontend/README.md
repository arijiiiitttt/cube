# Cube Frontend (Vite)

Modern Vite + React + TypeScript frontend for the **Cube** LLM Fine-Tuning Platform.

## Stack

- **Vite 5** + React 18 + TypeScript
- **React Router 6**
- **Tailwind CSS 3**
- **Recharts** (loss charts)
- **Lucide React** (icons)
- **Axios**
- **react-hot-toast**, **react-dropzone**

## Quick start

```bash
# Install
npm install

# Dev server (http://localhost:3000)
npm run dev

# Production build
npm run build
npm run preview
```

Create a `.env` file:

```
VITE_API_URL=http://localhost:8000
```

## Pages

| Route | Description |
|-------|-------------|
| `/dashboard/upload` | Upload dataset files |
| `/dashboard/dataset` | Process / tokenize dataset |
| `/dashboard/model` | Choose base model + QLoRA config |
| `/dashboard/training` | Live loss chart + logs |
| `/dashboard/evaluation` | Perplexity + generation playground |
| `/dashboard/deploy` | API key + chat playground |

## Notes

- Designed to talk to the existing Cube backend (port 8000).
- No Next.js / next-auth dependency.
- Clean, modern UI with Inter + JetBrains Mono.
