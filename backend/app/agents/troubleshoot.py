from typing import Dict, Any, List, Optional
import json
from ..llm import get_llm
from ..vector_db import VectorSearchProvider


class TroubleshootAgent:
    """
    Provides stepwise troubleshooting flows from mocked data.
    """

    def __init__(self, troubleshooting_data: List[Dict], vector_db: Optional[VectorSearchProvider] = None):
        self.troubleshooting = troubleshooting_data
        self.vector_db = vector_db
        self.llm = get_llm()

    async def diagnose(self, entities: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        appliance = entities.get("appliance_type")
        symptom = entities.get("symptom") or entities.get("symptom_slug")

        candidates = self._find_candidates(user_message, appliance, symptom)
        if not candidates:
            return {
                "response": "I couldn't find a troubleshooting guide for that issue. Tell me the appliance type and symptom (e.g., 'dishwasher not draining'). I can help with refrigerators and dishwashers.",
                "steps": [],
                "products": [],
                "metadata": {},
            }

        llm_struct = await self._llm_response(user_message, candidates)
        if llm_struct:
            return llm_struct

        # Fallback to deterministic formatting with first candidate
        return self._fallback_response(candidates[0])

    def _find_candidates(self, user_message: str, appliance: Optional[str], symptom: Optional[str]) -> List[Dict[str, Any]]:
        # First try vector search to rank, then map back to full records by symptom slug/display
        ranked = []
        if self.vector_db and self.vector_db.enabled:
            hits = self.vector_db.search_troubleshooting(
                query=user_message,
                appliance_type=appliance,
                top_k=5,
            )
            slug_to_obj = {t.get("symptom_slug") or t.get("symptom_display"): t for t in self.troubleshooting}
            for h in hits or []:
                key = h.get("symptom")
                if key in slug_to_obj:
                    ranked.append(slug_to_obj[key])

        # If still empty, filter by appliance/symptom substring
        if not ranked:
            for t in self.troubleshooting:
                if appliance and t.get("appliance_type") != appliance:
                    continue
                if symptom:
                    if symptom == t.get("symptom_slug") or symptom == t.get("symptom_display"):
                        ranked.append(t)
                        continue
                    if symptom.lower() in (t.get("symptom_display") or "").lower():
                        ranked.append(t)
                        continue
            if not ranked:
                ranked = [t for t in self.troubleshooting if not appliance or t.get("appliance_type") == appliance]

        # keep top 3 distinct entries
        seen = set()
        uniq = []
        for t in ranked:
            key = t.get("symptom_slug") or t.get("symptom_display")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        return uniq[:3]

    async def _llm_response(self, user_message: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        prompt = f"""You are a repair assistant. The user asked: \"{user_message}\".
You have up to 3 troubleshooting entries (JSON). Use them to craft a concise, helpful response.
Return STRICT JSON with keys: message (3-5 sentences), steps (list of {{step, detail, safety}}), metadata (object with common_causes, tags, about_repair, primary_component, replacement, repair_paths, clarifying_questions, source).

Entries:
{json.dumps(candidates[:3], indent=2)}
Rules:
- Focus on the most relevant entry; include up to 3 likely components from all candidates.
- Steps: include safety notes first, then diagnostic steps for the top component.
- Keep message brief; mention difficulty or story count if provided.
- Do not include markdown."""
        try:
            raw = await self.llm.generate(prompt, temperature=0.4, max_tokens=400, response_format=None)
            parsed = json.loads(raw)
            # minimal validation
            if isinstance(parsed, dict) and "message" in parsed:
                parsed.setdefault("products", [])
                parsed.setdefault("steps", [])
                parsed.setdefault("metadata", {})
                return parsed
        except Exception:
            return None
        return None

    def _fallback_response(self, match: Dict[str, Any]) -> Dict[str, Any]:
        # Reuse previous deterministic formatting for robustness
        appliance = match.get("appliance_type")
        symptom = match.get("symptom_slug") or match.get("symptom_display") or "Issue"
        title = match.get("symptom_display") or symptom or "Issue"
        summary = match.get("summary") or "Let's walk through the most common fixes."
        about = match.get("about_repair", {}) or {}
        difficulty = about.get("difficulty")
        story_ct = about.get("repair_stories_count")
        video_ct = about.get("step_by_step_videos_count")
        stats_bits = []
        if difficulty:
            stats_bits.append(f"Difficulty: {difficulty.title()}")
        if story_ct:
            stats_bits.append(f"{story_ct} owner fixes logged")
        if video_ct:
            stats_bits.append(f"{video_ct} video guides available")
        stats_line = " • ".join(stats_bits)

        top_path = None
        paths = match.get("repair_paths", [])
        if paths:
            top_path = sorted(paths, key=lambda p: p.get("path_rank", 999))[:1][0]
        cause_lines = []
        for p in sorted(paths, key=lambda p: p.get("path_rank", 999))[:3]:
            comp = p.get("component")
            why = p.get("why_it_causes_symptom")
            if comp and why:
                cause_lines.append(f"{comp}: {why}")

        bullets = []
        if top_path:
            component = top_path.get("component")
            why = top_path.get("why_it_causes_symptom")
            if component and why:
                bullets.append(f"Likely cause: {component} — {why}")
            replacement = top_path.get("replacement") or {}
            if replacement.get("category_label") and replacement.get("category_url"):
                bullets.append(f"Parts: {replacement['category_label']} ({replacement['category_url']})")
        if cause_lines:
            bullets.append("Other possible causes: " + " | ".join(cause_lines))

        cq = match.get("clarifying_questions") or []
        cq_line = f"Question to narrow it down: {cq[0]}" if cq else ""

        response_lines = [
            f"{title} on your {appliance}:",
            summary,
        ]
        if stats_line:
            response_lines.append(stats_line)
        response_lines.extend(bullets)
        if cq_line:
            response_lines.append(cq_line)
        response = "\n".join(response_lines)

        steps: List[Dict[str, Any]] = []
        if top_path:
            safety_notes = (top_path.get("diagnostic") or {}).get("safety_notes", [])
            for note in safety_notes:
                steps.append({"step": len(steps) + 1, "title": "Safety", "detail": note, "safety": True})
            diag_steps = (top_path.get("diagnostic") or {}).get("steps", []) or []
            for ds in diag_steps:
                steps.append({"step": len(steps) + 1, "title": top_path.get("component"), "detail": ds.get("detail")})

        causes = match.get("common_causes", [])
        tags = match.get("tags", [])
        replacement = (top_path or {}).get("replacement") if top_path else {}

        return {
            "response": response,
            "steps": steps,
            "metadata": {
                "common_causes": causes,
                "tags": tags,
                "about_repair": about,
                "primary_component": (top_path or {}).get("component"),
                "replacement": replacement,
                "repair_paths": paths,
                "clarifying_questions": cq,
                "source": match.get("source"),
            },
            "products": [],
        }
