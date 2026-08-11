"""
PTM-CoScientist Web UI (Streamlit)

테스트용 Web UI: 가설 생성, 토너먼트 결과, 실험 설계, Scientist-in-the-loop 피드백을 시각적으로 제공.
추후 PTM-platform 프론트엔드에 통합될 예정.
"""

import os

import requests
import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PTM-CoScientist",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API Base URL ─────────────────────────────────────────────────────────────
API_BASE = os.getenv("COSCIENTIST_API_URL", "http://localhost:8080")


# ─── Helper Functions ─────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Co-Scientist API 서버에 연결할 수 없습니다. `coscientist serve` 또는 Docker가 실행 중인지 확인하세요.")
        return None
    except Exception as e:  # noqa: BLE001 - UI must render API failures instead of crashing
        st.error(f"❌ API 오류: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Co-Scientist API 서버에 연결할 수 없습니다.")
        return None
    except Exception as e:  # noqa: BLE001 - UI must render API failures instead of crashing
        st.error(f"❌ API 오류: {e}")
        return None


def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001 - sidebar health check should degrade gracefully
        return False


def elo_color(elo: int) -> str:
    if elo >= 1600:
        return "🟢"
    elif elo >= 1500:
        return "🟡"
    else:
        return "🔴"


def priority_badge(priority: str) -> str:
    badges = {"high": "🔴 HIGH", "medium": "🟡 MEDIUM", "low": "🟢 LOW"}
    return badges.get(priority.lower(), priority.upper())


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("PTM-CoScientist")
    st.caption("AI Research Collaboration Agent")

    st.divider()

    # API Health Check
    if check_api_health():
        st.success("✅ API 서버 연결됨")
    else:
        st.error("❌ API 서버 오프라인")
        st.info("터미널에서 `coscientist serve` 를 실행하거나 Docker를 시작하세요.")

    st.divider()

    # Session Management
    st.subheader("🔖 세션 관리")
    if "session_id" in st.session_state and st.session_state.session_id:
        st.success(f"현재 세션: `{st.session_state.session_id}`")
        if st.button("🗑️ 세션 초기화", use_container_width=True):
            for key in ["session_id", "session_data", "feedback_history"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.info("아직 실행된 세션이 없습니다.")

    st.divider()
    st.caption("v0.1.0 — 테스트 빌드")
    st.caption("추후 PTM-platform에 통합 예정")


# ─── Main Content ─────────────────────────────────────────────────────────────

st.title("🧬 PTM-CoScientist")
st.markdown("**AI Co-Scientist for Post-Translational Modification Research** — Generate · Debate · Evolve")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 파이프라인 실행",
    "📊 가설 & 토너먼트 결과",
    "🔬 실험 설계",
    "💬 Scientist Feedback",
    "🧠 Scientific Reasoning",
])


# ─── Tab 1: Pipeline Run ──────────────────────────────────────────────────────

with tab1:
    st.header("파이프라인 실행")
    st.markdown("PTM-platform 분석이 완료된 Order를 입력하면, Co-Scientist가 가설을 생성하고 실험 설계를 제안합니다.")

    col1, col2 = st.columns([2, 1])

    with col1:
        order_code = st.text_input(
            "Order Code",
            placeholder="예: ORDER_20240701_001",
            help="PTM-platform에서 분석이 완료된 Order의 코드를 입력하세요.",
        )
        research_goal = st.text_area(
            "연구 목표 (Research Goal)",
            placeholder="예: 간 섬유화와 관련된 새로운 치료 표적을 발굴하고 싶습니다.",
            height=100,
            help="자연어로 연구 목표를 입력하면 AI가 이를 반영하여 가설을 생성합니다. 비워두면 데이터 기반으로 자동 생성합니다.",
        )

    with col2:
        ptm_type = st.selectbox(
            "PTM 유형",
            ["phosphorylation", "ubiquitylation"],
            help="분석할 PTM 유형을 선택하세요.",
        )
        iterations = st.slider(
            "반복 횟수 (Iterations)",
            min_value=1,
            max_value=5,
            value=3,
            help="Generate → Debate → Evolve 루프를 몇 번 반복할지 설정합니다. 횟수가 많을수록 가설 품질이 향상되지만 시간이 더 걸립니다.",
        )

    st.divider()

    col_run, col_status = st.columns([1, 2])

    with col_run:
        run_clicked = st.button(
            "▶️ 파이프라인 실행",
            type="primary",
            use_container_width=True,
            disabled=not order_code,
        )

    if run_clicked and order_code:
        with st.spinner("파이프라인 시작 중..."):
            result = api_post("/run", {
                "order_code": order_code,
                "research_goal": research_goal,
                "ptm_type": ptm_type,
                "max_iterations": iterations,
            })

        if result:
            session_id = result.get("session_id")
            st.session_state.session_id = session_id
            st.session_state.feedback_history = []
            st.success(f"✅ 파이프라인 시작됨! 세션 ID: `{session_id}`")
            st.info("'가설 & 토너먼트 결과' 탭에서 진행 상황을 확인하세요.")

    # Show current session info
    if "session_id" in st.session_state:
        st.divider()
        st.subheader("📡 현재 세션 상태 조회")

        if st.button("🔄 상태 새로고침", use_container_width=False):
            session_data = api_get(f"/session/{st.session_state.session_id}")
            if session_data:
                st.session_state.session_data = session_data

        if "session_data" in st.session_state:
            data = st.session_state.session_data
            status = data.get("status", "unknown")

            status_icon = {"completed": "✅", "running": "⏳", "started": "🚀"}.get(status, "❓")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("상태", f"{status_icon} {status.upper()}")
            col_b.metric("반복 완료", data.get("iteration", 0))
            col_c.metric("생성된 가설 수", data.get("total_hypotheses", 0))


# ─── Tab 2: Hypotheses & Tournament ──────────────────────────────────────────

with tab2:
    st.header("가설 & 토너먼트 결과")

    if "session_id" not in st.session_state:
        st.info("먼저 '파이프라인 실행' 탭에서 분석을 시작하세요.")
    else:
        if st.button("🔄 결과 불러오기", type="primary"):
            session_data = api_get(f"/session/{st.session_state.session_id}")
            if session_data:
                st.session_state.session_data = session_data

        if "session_data" in st.session_state:
            data = st.session_state.session_data
            hypotheses = data.get("top_hypotheses", [])

            if not hypotheses:
                st.warning("아직 생성된 가설이 없습니다. 파이프라인이 완료될 때까지 기다리거나 새로고침하세요.")
            else:
                st.success(f"총 **{data.get('total_hypotheses', 0)}개** 가설 생성됨 | 상위 {len(hypotheses)}개 표시")

                # Elo Distribution Chart
                st.subheader("📈 Elo 레이팅 분포")
                elo_data = {f"H{i+1} ({h.get('id', '')})": h.get("elo_rating", 1500)
                            for i, h in enumerate(hypotheses)}
                st.bar_chart(elo_data, color="#991B1B")

                st.divider()

                # Hypothesis Cards
                st.subheader("🏆 가설 목록 (Elo 순위)")

                for i, h in enumerate(hypotheses):
                    elo = h.get("elo_rating", 1500)
                    conf = h.get("confidence", 0.5)
                    category = h.get("category", "mechanistic")
                    status = h.get("status", "generated")

                    with st.expander(
                        f"{elo_color(elo)} **#{i+1}** | Elo: {elo} | {category.upper()} | {h.get('condition', '')[:80]}...",
                        expanded=(i == 0),
                    ):
                        col_left, col_right = st.columns([3, 1])

                        with col_left:
                            st.markdown(f"**IF:** {h.get('condition', '')}")
                            st.markdown(f"**THEN:** {h.get('prediction', '')}")
                            st.markdown(f"**BECAUSE:** {h.get('mechanism', '')}")
                            if h.get("signaling_chain"):
                                st.markdown(f"**Signaling:** `{h.get('signaling_chain', '')}`")
                            if h.get("testable_prediction"):
                                st.info(f"🔬 **Testable Prediction:** {h.get('testable_prediction', '')}")

                        with col_right:
                            st.metric("Elo Rating", elo)
                            st.metric("Confidence", f"{conf:.0%}")
                            st.metric("Status", status.upper())
                            if h.get("supporting_ptms"):
                                st.markdown("**Supporting PTMs:**")
                                for ptm in h.get("supporting_ptms", [])[:5]:
                                    st.markdown(f"- `{ptm}`")

                        # Evidence
                        ev_for = h.get("evidence_for", [])
                        ev_against = h.get("evidence_against", [])
                        if ev_for or ev_against:
                            st.divider()
                            col_ev1, col_ev2 = st.columns(2)
                            with col_ev1:
                                st.markdown(f"✅ **지지 근거** ({len(ev_for)}건)")
                                for ev in ev_for[:2]:
                                    title = ev.get("title") or ev.get("source") or "Unknown"
                                    excerpt = ev.get("excerpt") or ev.get("text") or ""
                                    st.caption(f"📄 {title}: {excerpt[:120]}...")
                            with col_ev2:
                                st.markdown(f"❌ **반박 근거** ({len(ev_against)}건)")
                                for ev in ev_against[:2]:
                                    title = ev.get("title") or ev.get("source") or "Unknown"
                                    excerpt = ev.get("excerpt") or ev.get("text") or ""
                                    st.caption(f"📄 {title}: {excerpt[:120]}...")

                        reflection = h.get("reflection", {})
                        if reflection:
                            st.divider()
                            st.markdown("🪞 **Self-Critique / Reflection**")
                            st.caption(reflection.get("summary", ""))
                            col_ref1, col_ref2 = st.columns(2)
                            with col_ref1:
                                st.markdown(f"**Data consistency:** `{reflection.get('data_consistency', 'N/A')}`")
                                for item in reflection.get("confounders", [])[:3]:
                                    st.markdown(f"- Confounder: {item}")
                            with col_ref2:
                                st.markdown(f"**Recommended action:** `{reflection.get('recommended_action', 'N/A')}`")
                                for item in reflection.get("falsification_conditions", [])[:3]:
                                    st.markdown(f"- Falsify if: {item}")

                        # Debate History
                        debate = h.get("debate_history", [])
                        if debate:
                            st.divider()
                            st.markdown(f"⚔️ **토너먼트 기록** ({len(debate)}전)")
                            wins = sum(1 for d in debate if d.get("result") == "win")
                            losses = sum(1 for d in debate if d.get("result") == "loss")
                            draws = sum(1 for d in debate if d.get("result") == "draw")
                            st.markdown(f"🏅 {wins}승 {losses}패 {draws}무")


# ─── Tab 3: Experiment Designs ────────────────────────────────────────────────

with tab3:
    st.header("실험 설계")
    st.markdown("상위 가설에 대한 구체적인 실험 프로토콜을 자동으로 생성합니다.")

    if "session_id" not in st.session_state:
        st.info("먼저 '파이프라인 실행' 탭에서 분석을 시작하세요.")
    else:
        col_gen, col_n = st.columns([2, 1])
        with col_n:
            top_n = st.number_input("상위 N개 가설에 대해 설계", min_value=1, max_value=10, value=5)

        with col_gen:
            if st.button("🔬 실험 설계 생성", type="primary"):
                with st.spinner("실험 프로토콜 생성 중..."):
                    result = api_post(
                        f"/session/{st.session_state.session_id}/design-experiments",
                        {},
                    )
                    if result:
                        st.session_state.session_data = st.session_state.get("session_data", {})
                        st.session_state.session_data["experiment_designs"] = result.get("designs", [])
                        st.success(f"✅ {len(result.get('designs', []))}개 실험 설계 완료!")

        if "session_data" in st.session_state:
            designs = st.session_state.session_data.get("experiment_designs", [])

            if designs:
                st.divider()
                for i, d in enumerate(designs):
                    priority = d.get("priority", "medium")
                    badge = priority_badge(priority)

                    with st.expander(
                        f"{badge} | **{d.get('title', f'Experiment {i+1}')}** | {d.get('approach', '')}",
                        expanded=(i == 0),
                    ):
                        col_l, col_r = st.columns([2, 1])

                        with col_l:
                            st.markdown(f"**목적:** {d.get('objective', '')}")
                            st.markdown(f"**접근법:** `{d.get('approach', '')}`")
                            st.markdown(f"**예상 결과:** {d.get('expected_outcome', '')}")
                            st.markdown(f"**대안 결과:** {d.get('alternative_outcome', '')}")
                            if d.get("rationale"):
                                st.info(f"💡 **선택 이유:** {d.get('rationale', '')}")

                        with col_r:
                            st.metric("우선순위", badge)
                            st.metric("예상 기간", d.get("estimated_timeline", "N/A"))

                            if d.get("key_reagents"):
                                st.markdown("**주요 시약:**")
                                for r in d.get("key_reagents", []):
                                    st.markdown(f"- {r}")

                            if d.get("controls"):
                                st.markdown("**대조군:**")
                                for c in d.get("controls", []):
                                    st.markdown(f"- {c}")

                        st.caption(f"🔗 가설 ID: `{d.get('hypothesis_id', 'N/A')}`")


# ─── Tab 4: Scientist Feedback ────────────────────────────────────────────────

with tab4:
    st.header("💬 Scientist-in-the-Loop Feedback")
    st.markdown(
        "AI가 생성한 가설의 방향을 연구자가 직접 조정할 수 있습니다. "
        "피드백을 제출한 후 파이프라인을 재실행하면 다음 이터레이션에 반영됩니다."
    )

    if "session_id" not in st.session_state:
        st.info("먼저 '파이프라인 실행' 탭에서 분석을 시작하세요.")
    else:
        # Feedback input
        st.subheader("피드백 입력")

        feedback_type = st.selectbox(
            "피드백 유형",
            options=["direction", "constraint", "seed_idea"],
            format_func=lambda x: {
                "direction": "🧭 방향 제시 (Direction) — 특정 분야나 표적에 집중하도록 유도",
                "constraint": "🚫 제약 조건 (Constraint) — 특정 메커니즘이나 경로를 제외",
                "seed_idea": "💡 초기 아이디어 (Seed Idea) — 탐색할 특정 가설 방향 제안",
            }[x],
        )

        feedback_examples = {
            "direction": "예: 간 섬유화와 관련된 치료 표적에 집중해주세요.",
            "constraint": "예: 유비퀴틴 매개 분해 경로는 제외해주세요.",
            "seed_idea": "예: EGFR과 Hippo 경로 간의 크로스토크를 탐색해주세요.",
        }

        feedback_content = st.text_area(
            "피드백 내용",
            placeholder=feedback_examples[feedback_type],
            height=100,
        )

        if st.button("📤 피드백 제출", type="primary", disabled=not feedback_content):
            result = api_post(
                f"/session/{st.session_state.session_id}/feedback",
                {
                    "session_id": st.session_state.session_id,
                    "feedback_type": feedback_type,
                    "content": feedback_content,
                },
            )
            if result:
                if "feedback_history" not in st.session_state:
                    st.session_state.feedback_history = []
                st.session_state.feedback_history.append({
                    "type": feedback_type,
                    "content": feedback_content,
                })
                st.success(f"✅ 피드백이 제출되었습니다. (총 {result.get('total_feedback', 0)}개)")
                st.info("'파이프라인 실행' 탭에서 파이프라인을 재실행하면 이 피드백이 다음 이터레이션에 반영됩니다.")

        # Feedback History
        if st.session_state.get("feedback_history"):
            st.divider()
            st.subheader("📋 제출된 피드백 목록")
            type_icons = {"direction": "🧭", "constraint": "🚫", "seed_idea": "💡"}
            for i, fb in enumerate(st.session_state.feedback_history):
                icon = type_icons.get(fb["type"], "💬")
                st.markdown(f"**{i+1}.** {icon} `{fb['type'].upper()}` — {fb['content']}")


# ─── Tab 5: Scientific Reasoning ───────────────────────────────────────────────

with tab5:
    st.header("🧠 Scientific Reasoning")
    st.markdown(
        "관찰 데이터, 문헌 근거, Self-Critique, 가설 다양성, Meta-review, "
        "그리고 연구자가 입력한 실험 결과를 하나의 audit trail로 확인합니다."
    )

    if "session_id" not in st.session_state:
        st.info("먼저 '파이프라인 실행' 탭에서 분석을 시작하세요.")
    else:
        if st.button("🔄 Scientific Reasoning 불러오기", type="primary"):
            reasoning = api_get(f"/session/{st.session_state.session_id}/scientific-reasoning")
            if reasoning:
                st.session_state.reasoning_data = reasoning
            session_data = api_get(f"/session/{st.session_state.session_id}")
            if session_data:
                st.session_state.session_data = session_data

        reasoning = st.session_state.get("reasoning_data", {})
        if reasoning:
            graph = reasoning.get("evidence_graph", {})
            summary = graph.get("summary", {})
            st.subheader("Evidence Graph")
            col1, col2, col3 = st.columns(3)
            col1.metric("Nodes", summary.get("node_count", 0))
            col2.metric("Edges", summary.get("edge_count", 0))
            col3.metric("Relations", len(summary.get("relations", {})))
            with st.expander("Graph summary", expanded=False):
                st.json(summary)

            diversity = reasoning.get("diversity_summary", {})
            if diversity:
                st.subheader("Proximity & Diversity")
                st.caption(
                    f"Method: `{diversity.get('method', 'N/A')}` | "
                    f"Clusters: {diversity.get('cluster_count', 0)}"
                )
                for cluster in diversity.get("clusters", []):
                    st.markdown(
                        f"- `{cluster.get('id', '')}`: representative "
                        f"`{cluster.get('representative_hypothesis_id', '')}` | "
                        f"members: {len(cluster.get('member_hypothesis_ids', []))}"
                    )

            meta_review = reasoning.get("meta_review", {})
            if meta_review:
                st.subheader("Meta-review")
                st.info(meta_review.get("executive_summary", ""))
                leading = meta_review.get("leading_mechanism", {})
                if leading:
                    st.markdown(
                        f"**Leading candidate:** `{leading.get('hypothesis_id', '')}` — "
                        f"{leading.get('rationale', '')}"
                    )
                if meta_review.get("key_uncertainties"):
                    st.markdown("**Key uncertainties**")
                    for item in meta_review["key_uncertainties"]:
                        st.markdown(f"- {item}")

            reflections = reasoning.get("hypothesis_reflections", [])
            if reflections:
                with st.expander("Reflection records", expanded=False):
                    st.json(reflections)

        st.divider()
        st.subheader("Lab-in-the-loop: 실험 결과 기록")
        st.caption(
            "입력한 결과는 즉시 Evidence Graph에 provenance로 저장됩니다. "
            "가설의 확정적 증명으로 처리되지 않으며, 재실행 시 Reflection·Debate의 근거로 사용됩니다."
        )
        hypotheses = st.session_state.get("session_data", {}).get("top_hypotheses", [])
        if not hypotheses:
            st.info("결과를 불러온 후 가설을 선택할 수 있습니다.")
        else:
            label_to_id = {
                f"{hypothesis.get('id', '')} | {hypothesis.get('prediction', '')[:70]}": hypothesis.get("id", "")
                for hypothesis in hypotheses
            }
            with st.form("lab_result_form"):
                selected_label = st.selectbox("검증한 가설", options=list(label_to_id))
                outcome = st.selectbox(
                    "실험 결과",
                    options=["supports", "contradicts", "inconclusive"],
                    format_func=lambda value: {
                        "supports": "Supports — 가설과 일치",
                        "contradicts": "Contradicts — 가설과 상충",
                        "inconclusive": "Inconclusive — 판단 불가",
                    }[value],
                )
                assay_type = st.text_input("Assay / Technique", placeholder="예: phospho-Western blot")
                result_summary = st.text_area("결과 요약", placeholder="관찰한 정량적·정성적 결과를 기록하세요.")
                observed_effect = st.text_input("관찰된 효과", placeholder="예: EGFR inhibitor 처리 시 SRC-Y416 신호 감소")
                controls_text = st.text_input("대조군", placeholder="예: vehicle, untreated, kinase-dead mutant")
                source_reference = st.text_input("원자료 참조", placeholder="예: 실험 노트 ID, 파일명, 내부 보고서 링크")
                submitted = st.form_submit_button("🧪 실험 결과 기록")

            if submitted:
                payload = {
                    "hypothesis_id": label_to_id[selected_label],
                    "outcome": outcome,
                    "assay_type": assay_type,
                    "result_summary": result_summary,
                    "observed_effect": observed_effect,
                    "controls": [item.strip() for item in controls_text.split(",") if item.strip()],
                    "source_reference": source_reference,
                }
                result = api_post(f"/session/{st.session_state.session_id}/lab-results", payload)
                if result:
                    st.success("✅ 실험 결과가 기록되었습니다.")
                    st.info("이 결과를 Reflection·Debate·Ranking에 반영하려면 Scientist Feedback 탭에서 재실행하세요.")
