"""Independently reconstruct episode money, inventory and aggregate impact."""
import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
from pathlib import Path
import pandas as pd


def audit_ledger(rows, *, initial_cash, initial_log_odds, beta, inventory_limit,
                 maker_fee_multiplier, balance_precision, quantity_lot, participation):
    cash, q, latent = Decimal(str(initial_cash)), Decimal(0), float(initial_log_odds)
    grid, lot = Decimal(balance_precision), Decimal(quantity_lot)
    orders, accumulators, fills, trades, settlements, total_fees = {}, {}, 0, 0, 0, Decimal(0)
    previous = None
    for index,row in enumerate(rows):
        kind = row["kind"]
        if kind=="quote":
            key = str(row["order_id"])
            orders[key] = Decimal(str(row["quantity"]))
            accumulators[key] = Decimal(0)
        elif kind=="cancel":
            orders.pop(str(row["order_id"]), None)
        elif kind=="noise":
            latent += float(row["innovation_logodds"])
        elif kind=="market":
            trades += 1
            volume, own = float(row["market_quantity"]), float(row["maker_quantity"])
            sign = int(Decimal(str(row["market_sign"])))
            assert sign in (-1,1) and 0<=own<=volume*participation+1e-10
            if own:
                assert previous["kind"]=="fill"
                assert float(previous["quantity"])==own
                assert previous["side"]==("ask" if sign==1 else "bid")
            impact = beta*sign*volume
            assert abs(float(row["impact_logodds"])-impact)<1e-10
            latent += impact
        elif kind=="fill":
            fills += 1
            key = str(row["order_id"])
            quantity, price = Decimal(str(row["quantity"])), Decimal(str(row["price"]))
            assert 0<quantity<=orders[key] and quantity%lot==0
            orders[key] -= quantity
            side = row["side"]
            revenue = quantity*price*(-1 if side=="bid" else 1)
            fee = Decimal(str(row["fee"]))
            model = Decimal(".0175")*Decimal(str(maker_fee_multiplier))*quantity*price*(1-price)
            trade = model.quantize(Decimal(".000001"), rounding=ROUND_CEILING)
            aligned = ((revenue-trade)/grid).to_integral_value(rounding=ROUND_FLOOR)*grid
            rounding = revenue-trade-aligned
            accumulators[key] += rounding
            rebate = min((accumulators[key]/grid).to_integral_value(rounding=ROUND_FLOOR)*grid,
                ((trade+rounding)/grid).to_integral_value(rounding=ROUND_FLOOR)*grid)
            accumulators[key] -= rebate
            assert fee==trade+rounding-rebate and fee>=0
            assert Decimal(str(row["trade_fee"]))==trade
            assert Decimal(str(row["rounding_fee"]))==rounding
            assert Decimal(str(row["rebate"]))==rebate
            assert Decimal(str(row["cash_change"]))==revenue-fee
            assert Decimal(str(row["cash_before"]))==cash and Decimal(str(row["inventory_before"]))==q
            total_fees += fee
            cash += revenue-fee
            q += quantity*(1 if side=="bid" else -1)
            limit = Decimal(str(row["execution_limit"]))
            assert price>=limit if side=="bid" else price<=limit
        elif kind=="settlement":
            settlements += 1
            outcome = int(Decimal(str(row["settlement_outcome"])))
            assert settlements==1 and outcome in (0,1)
            adjustment = q*outcome
            assert Decimal(str(row["settlement_adjustment"]))==adjustment
            assert Decimal(str(row["terminal_inventory"]))==q
            cash += adjustment
            q = Decimal(0)
        else:
            raise AssertionError(f"Unknown ledger event: {kind}")
        assert Decimal(str(row["cash"]))==cash, f"Cash mismatch at row {index}"
        assert Decimal(str(row["inventory"]))==q, f"Inventory mismatch at row {index}"
        assert cash>=max(-q,Decimal(0)) and abs(q)<=Decimal(str(inventory_limit))
        assert abs(float(row["log_odds"])-latent)<1e-8, f"Latent price mismatch at row {index}"
        previous = row
    return {"passed": True, "ledger_rows": len(rows), "market_trades": trades, "maker_fills": fills,
        "settlements": settlements, "terminal_cash": str(cash), "terminal_inventory": str(q),
        "fees": str(total_fees), "terminal_log_odds": latent}


def validate_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    results = []
    for episode in manifest["episodes"]:
        # Read as strings to preserve money, not binary floating-point round trips.
        rows = pd.read_csv(directory/episode["ledger_file"],dtype=str).to_dict("records")
        report = audit_ledger(rows, **episode["audit_arguments"])
        steps = pd.read_csv(directory/episode["steps_file"])
        equity_change = float(report["terminal_cash"])-episode["audit_arguments"]["initial_cash"]
        if report["settlements"]==0:
            equity_change += float(report["terminal_inventory"])*steps.iloc[-1].probability
        assert abs(float(steps.reward.sum()+steps.risk_penalty.sum())-equity_change)<1e-8
        assert report["maker_fills"]==episode["maker_fills"]
        report["episode_id"] = episode["episode_id"]
        results.append(report)
    # Verify that the runnable code/config still matches the recorded run.
    from .market_dynamics import file_hash
    root = Path(__file__).resolve().parents[1]
    for relative, checksum in manifest["source_sha256"].items():
        assert file_hash(root/relative)==checksum, f"Run source changed: {relative}"
    for absolute, checksum in manifest["config_sha256"].items():
        assert file_hash(absolute)==checksum, f"Run config changed: {absolute}"
    result = {"passed": True, "episodes": len(results), "maker_fills": sum(r["maker_fills"] for r in results),
              "market_trades": sum(r["market_trades"] for r in results), "checks": results}
    (directory/"validation_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = validate_run(parser.parse_args().directory)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))
