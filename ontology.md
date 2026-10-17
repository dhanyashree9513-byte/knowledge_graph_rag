# Ontology

**Corpus:** latest 10-K of NVIDIA, AMD and Intel. Item 1 (Business) and Item 1A (Risk Factors) only.

The extractor may only emit the types below. Anything else is dropped.

## Entity types

| Type | Definition | Examples |
|---|---|---|
| Company | A business named in a filing | NVIDIA, TSMC, Microsoft |
| Product | A named chip, platform or product line | H100, EPYC, Xeon |
| Market | An end market or segment a company sells into | Data Center, Gaming, PC |
| Technology | A process node, architecture or standard | CUDA, 3nm, x86 |
| Risk | A named risk factor | Export controls, supply concentration |
| Location | A country or region that matters to the business | Taiwan, China |
| Person | An executive named in the filing | Jensen Huang, Lisa Su |

## Relationship types

| Relationship | Applies when |
|---|---|
| (Company)-[SUPPLIES]->(Company) | Supplier → buyer: the first makes or sells components or services for the second. Also covers "we buy from X" (no separate CUSTOMER_OF, so one fact has one form) |
| (Company)-[COMPETES_WITH]->(Company) | The filing names the other company as a competitor |
| (Company)-[MAKES]->(Product) | The company sells the product |
| (Product)-[TARGETS]->(Market) | The product is sold into the market |
| (Product)-[USES]->(Technology) | The product is built on the technology |
| (Company)-[FACES]->(Risk) | The filing lists the risk |
| (Risk)-[INVOLVES]->(Company \| Location) | The risk is about that supplier or country |
| (Company)-[OPERATES_IN]->(Location) | The company has manufacturing or significant sales there |
| (Person)-[EXECUTIVE_OF]->(Company) | The person is a named officer |

## Driving questions (multi-hop)

| # | Question | Path |
|---|---|---|
| 1 | Which suppliers do NVIDIA and AMD share? | (NVIDIA)<-[SUPPLIES]-(s:Company)-[SUPPLIES]->(AMD) |
| 2 | Which risks are faced by every company that TSMC supplies? | (TSMC)-[SUPPLIES]->(c)-[FACES]->(r:Risk) |
| 3 | Which markets do NVIDIA's competitors target with their products? | (NVIDIA)-[COMPETES_WITH]->(c)-[MAKES]->(p)-[TARGETS]->(m:Market) |
| 4 | Which locations are involved in risks faced by Intel's competitors? | (Intel)-[COMPETES_WITH]->(c)-[FACES]->(r)-[INVOLVES]->(l:Location) |
| 5 | Which technologies are shared by products targeting the Data Center market? | (m:Market {name:"Data Center"})<-[TARGETS]-(p)-[USES]->(t:Technology) |

## Out of scope
- Financial figures (revenue, margins): numbers belong in a table, not a graph, and the 10-K tables parse badly.
- Legal proceedings and board members: no driving question needs them.
- Items 7 and 8 (MD&A, financial statements): long, numeric, and expensive to extract.
