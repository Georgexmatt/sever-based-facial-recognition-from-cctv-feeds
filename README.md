# Server-Based Facial Recognition from CCTV Feeds

Production-style Python 3.12 architecture for a server-based facial recognition attendance system that separates edge video processing from server-side identity matching and attendance persistence.

## Project Structure

```text
.
├── configs/                         # Environment-specific configuration templates and runtime config files loaded through environment variables.
├── data/                            # Local development data mount points; never hardcode these paths in code.
│   ├── faiss/                       # FAISS index artifacts for initial vector search experiments.
│   ├── gallery/                     # Local face gallery assets used during development and controlled testing.
│   ├── logs/                        # Local Loguru log output when file logging is enabled.
│   ├── models/                      # Local ONNX/model artifact mount point for SCRFD, AdaFace, MiniFASNet, and related models.
│   └── sqlite/                      # Local SQLite database files for development and internship validation.
├── docs/                            # Product, architecture, operations, and API documentation.
│   ├── api/                         # FastAPI endpoint contracts, request/response examples, and integration notes.
│   ├── architecture/                # System diagrams, data-flow documentation, trade-offs, and design decisions.
│   └── operations/                  # Deployment, observability, model artifact management, and runbook documentation.
├── scripts/                         # Operational helper scripts for development, migrations, indexing, and maintenance.
├── src/                             # Application source code using a src-layout package structure.
│   └── face_attendance/             # Main Python package for the facial recognition attendance product.
│       ├── api/                     # FastAPI routers, dependencies, middleware, and HTTP error mapping.
│       │   └── v1/                  # Versioned API surface to keep future API revisions backward compatible.
│       ├── application/             # Use-case orchestration layer that coordinates domain services and infrastructure adapters.
│       │   ├── attendance/          # Attendance workflows, check-in/check-out policies, and event handling.
│       │   ├── edge/                # Edge pipeline orchestration for RTSP ingestion, detection, tracking, and best-frame selection.
│       │   └── server/              # Server pipeline orchestration for embeddings, matching, liveness, review, and calibration.
│       ├── core/                    # Cross-cutting application primitives such as settings, dependency injection, constants, and lifecycle wiring.
│       ├── domain/                  # Framework-independent business rules and interfaces.
│       │   ├── entities/            # Domain entities such as camera, face track, person, embedding, match, and attendance record.
│       │   ├── repositories/        # Repository interfaces for persistence and vector search abstractions.
│       │   └── services/            # Pure domain services for scoring, attendance decisions, and gallery policies.
│       ├── edge/                    # Edge-side computer vision components.
│       │   ├── capture/             # RTSP reader abstractions, frame sources, reconnect behavior, and stream metadata handling.
│       │   ├── detection/           # SCRFD face detection adapters and detector interface implementations.
│       │   ├── selection/           # Best-frame selection logic based on blur, pose, size, occlusion, and quality heuristics.
│       │   └── tracking/            # ByteTrack integration and track lifecycle management for faces across frames.
│       ├── infrastructure/          # External-system adapters and implementation details kept outside the domain layer.
│       │   ├── database/            # SQLAlchemy models, sessions, migrations, and SQLite-backed repository implementations.
│       │   ├── logging/             # Loguru configuration, structured logging helpers, and correlation/request context.
│       │   ├── model_runtime/       # ONNX Runtime session management, device/provider selection, and model loading utilities.
│       │   └── vector_stores/       # FAISS implementations now and Qdrant adapters later behind common repository interfaces.
│       ├── schemas/                 # Pydantic request/response DTOs and validation models for API and service boundaries.
│       ├── server/                  # Server-side machine learning and identity services.
│       │   ├── calibration/         # Threshold calibration workflows, evaluation metrics, and score distribution analysis.
│       │   ├── embedding/           # AdaFace embedding extraction adapters and normalization utilities.
│       │   ├── liveness/            # Future MiniFASNet anti-spoofing integration isolated from recognition logic.
│       │   ├── matching/            # Face search, candidate ranking, thresholding, and FAISS/Qdrant matching strategies.
│       │   └── review/              # Future review queue workflows for low-confidence matches and human verification.
│       └── utils/                   # Small reusable helpers that are not domain-specific and have no framework coupling.
├── tests/                           # Automated tests organized by scope.
│   ├── e2e/                         # End-to-end tests for realistic API and pipeline flows.
│   ├── integration/                 # Integration tests for database, vector store, model runtime, and API wiring.
│   └── unit/                        # Fast unit tests for domain logic, services, schemas, and utility functions.
├── main.py                          # Current empty compatibility entry point; FastAPI application wiring will move under src/face_attendance.
└── requirements.txt                 # Python dependency lock/input file for the current internship environment.
```

## Directory Design Principles

- `domain/` contains business concepts and interfaces only; it should not import FastAPI, OpenCV, SQLAlchemy, FAISS, ONNX Runtime, or vendor SDKs.
- `application/` coordinates use cases through dependency-injected interfaces so edge and server workflows remain testable.
- `infrastructure/` owns implementation details for databases, vector indexes, model runtimes, and logging providers.
- `edge/` contains computer-vision pipeline components that operate close to CCTV feeds before server recognition.
- `server/` contains identity, embedding, matching, liveness, and review capabilities that run centrally.
- `api/` exposes stable HTTP contracts and delegates business decisions to the application layer.
- `configs/` and environment variables should drive runtime behavior; model paths, database URLs, thresholds, RTSP URLs, and provider settings must not be hardcoded.