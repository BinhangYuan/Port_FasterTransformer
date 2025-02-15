import asyncio
import os
import sys
from enum import Enum
from dacite import from_dict 
from dataclasses import asdict
from typing import Dict
from loguru import logger
from typing import Any
import torch.distributed as dist


from .together_web3.computer import (
    Instance,
    Job,
    ImageModelInferenceResult,
    LanguageModelInferenceChoice,
    LanguageModelInferenceResult,
    MatchEvent,
    ResourceTypeInstance,
    Result,
    RequestTypeImageModelInference,
    RequestTypeLanguageModelInference,
    RequestTypeShutdown,
    ResultEnvelope,
)
from .together_web3.coordinator import Join, JoinEnvelope
from .together_web3.together import TogetherClientOptions, TogetherWeb3

class ServiceDomain(Enum):
    http = "http"
    together = "together"

class FastInferenceInterface:
    def dispatch_request(args: Dict[str, any], match_event: MatchEvent):
        raise NotImplementedError

    def __init__(self, model_name: str, args=None) -> None:
        self.model_name = model_name
        self.service_domain = args.get("service_domain", ServiceDomain.together)
        self.coordinator: TogetherWeb3 = args.get(
            "coordinator") if self.service_domain == ServiceDomain.together else None
        self.shutdown = False

    def start(self):
        loop = asyncio.get_event_loop()
        future = asyncio.Future()
        asyncio.ensure_future(self._run_together_server())
        loop.run_forever()
    
    def worker(self):
        pass

    async def _run_together_server(self) -> None:
        # only rank 0
        if dist.get_rank() == 0:
            self.coordinator._on_disconnect.append(self._join_local_coordinator)
            self.coordinator._on_match_event.append(self.together_request)
            await self._join_local_coordinator()
            logger.info("Start _run_together_server")
            try:
                while not self.shutdown:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f'_run_together_server failed: {e}')
            self._shutdown()
        else:
            self.worker()
        

    async def _join_local_coordinator(self):
        try:
            logger.info("_join_local_coordinator")
            join = Join(
                group_name="group1",
                worker_name="worker"+str(dist.get_rank()),
                host_name="",
                host_ip="",
                interface_ip=[],
                instance=Instance(
                    arch="",
                    os="",
                    cpu_num=0,
                    gpu_num=0,
                    gpu_type="",
                    gpu_memory=0,
                    resource_type=ResourceTypeInstance,
                    tags={}
                ),
                config={
                    "model": "opt66b",
                    "request_type": RequestTypeLanguageModelInference,
                },
            )
            self.subscription_id = await self.coordinator.get_subscription_id("coordinator")
            args = self.coordinator.coordinator.join(
                asdict(JoinEnvelope(join=join, signature=None)), self.subscription_id)

        except Exception as e:
            logger.exception(f'_join_local_coordinator failed: {e}')


    async def together_request(self, match_event: MatchEvent, raw_event: Dict[str, Any]) -> None:
        logger.info(f"together_request {raw_event}")
        request_json = [raw_event["match"]["service_bid"]["job"]]
        response_json = self.dispatch_request(request_json, match_event)
        await self.send_result_back(match_event, response_json)

    async def send_result_back(self, match_event: MatchEvent, result_data: Dict[str, Any], partial: bool = False) -> None:
        try:
            #logger.info(f"send_result_back {result_data}")
            result = {
                "ask_address": match_event.match.ask_address,
                "bid_address": match_event.match.bid_address,
                "ask_offer_id": match_event.match.ask_offer_id,
                "bid_offer_id": match_event.match.bid_offer_id,
                "match_id": match_event.match_id,
                "data": result_data,
            }
            if partial:
                result["partial"] = True
            await self.coordinator.update_result(ResultEnvelope(
                result=from_dict(
                    data_class=Result,
                    data=result,
                ),
                signature=None,
            ))
        except Exception as e:
            logger.error(f"send_result_back error: {e}")

    def _shutdown(self) -> None:
        logger.info("Shutting down")


#if __name__ == "__main__":
#    fip = FastInferenceInterface(model_name="opt66b")
#    fip.start()
