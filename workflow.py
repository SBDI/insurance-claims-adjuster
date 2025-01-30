from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step
)
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.core.retrievers import BaseRetriever
from schemas import ClaimInfo, PolicyQueries, PolicyRecommendation, ClaimDecision
from prompts import GENERATE_POLICY_QUERIES_PROMPT, POLICY_RECOMMENDATION_PROMPT
from config import LLM_MODEL_NAME
from utils import get_declarations_docs
from llama_index.core import VectorStoreIndex
import logging

logger = logging.getLogger(__name__)


class ClaimInfoEvent(Event):
    claim_info: ClaimInfo

class PolicyQueryEvent(Event):
    queries: PolicyQueries

class PolicyMatchedEvent(Event):
    policy_text: str

class RecommendationEvent(Event):
    recommendation: PolicyRecommendation

class DecisionEvent(Event):
    decision: ClaimDecision

class LogEvent(Event):
    msg: str
    delta: bool = False

class AutoInsuranceWorkflow(Workflow):
    def __init__(
        self,
        policy_index: VectorStoreIndex,
        declarations_index: VectorStoreIndex,
        llm: LLM | None = None,
        output_dir: str = "data_out",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.policy_retriever = policy_index.as_retriever(rerank_top_n=3)
        self.declarations_index = declarations_index
        self.llm = llm or TogetherLLM(model=LLM_MODEL_NAME)

    @step
    async def load_claim_info(self, ctx: Context, ev: StartEvent) -> ClaimInfoEvent:
        logger.info(">> Loading Claim Info")
        from utils import parse_claim
        claim_info = parse_claim(ev.claim_json_path)
        await ctx.set("claim_info", claim_info)
        return ClaimInfoEvent(claim_info=claim_info)

    @step
    async def generate_policy_queries(self, ctx: Context, ev: ClaimInfoEvent) -> PolicyQueryEvent:
        logger.info(">> Generating Policy Queries")
        prompt = ChatPromptTemplate.from_messages([("user", GENERATE_POLICY_QUERIES_PROMPT)])
        queries = await self.llm.astructured_predict(
            PolicyQueries,
            prompt,
            claim_info=ev.claim_info.model_dump_json()
        )
        return PolicyQueryEvent(queries=queries)

    @step
    async def retrieve_policy_text(self, ctx: Context, ev: PolicyQueryEvent) -> PolicyMatchedEvent:
        logger.info(">> Retrieving policy sections")
        claim_info = await ctx.get("claim_info")

        combined_docs = {}
        for query in ev.queries.queries:
             logger.info(f">> Query: {query}")
            # fetch policy text
             docs = await self.policy_retriever.aretrieve(query)
             for d in docs:
                combined_docs[d.id_] = d

        # also fetch the declarations page for the policy holder
        d_docs = get_declarations_docs(self.declarations_index, claim_info.policy_number)
        if d_docs: # handle cases where no metadata has been set
          d_doc = d_docs[0]
          combined_docs[d_doc.id_] = d_doc
        policy_text = "\n\n".join([doc.get_content() for doc in combined_docs.values()])
        await ctx.set("policy_text", policy_text)
        return PolicyMatchedEvent(policy_text=policy_text)

    @step
    async def generate_recommendation(self, ctx: Context, ev: PolicyMatchedEvent) -> RecommendationEvent:
        logger.info(">> Generating Policy Recommendation")
        claim_info = await ctx.get("claim_info")
        prompt = ChatPromptTemplate.from_messages([("user", POLICY_RECOMMENDATION_PROMPT)])
        recommendation = await self.llm.astructured_predict(
            PolicyRecommendation,
            prompt,
            claim_info=claim_info.model_dump_json(),
            policy_text=ev.policy_text
        )
        logger.info(f">> Recommendation: {recommendation.model_dump_json()}")
        return RecommendationEvent(recommendation=recommendation)

    @step
    async def finalize_decision(self, ctx: Context, ev: RecommendationEvent) -> DecisionEvent:
        logger.info(">> Finalizing Decision")
        claim_info = await ctx.get("claim_info")
        rec = ev.recommendation
        covered = "covered" in rec.recommendation_summary.lower() or (rec.settlement_amount is not None and rec.settlement_amount > 0)
        deductible = rec.deductible if rec.deductible is not None else 0.0
        recommended_payout = rec.settlement_amount if rec.settlement_amount else 0.0
        decision = ClaimDecision(
            claim_number=claim_info.claim_number,
            covered=covered,
            deductible=deductible,
            recommended_payout=recommended_payout,
            notes=rec.recommendation_summary
        )
        return DecisionEvent(decision=decision)

    @step
    async def output_result(self, ctx: Context, ev: DecisionEvent) -> StopEvent:
        logger.info(f">> Decision: {ev.decision.model_dump_json()}")
        return StopEvent(result={"decision": ev.decision})