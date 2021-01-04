import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)
    
    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    p = dict()
    zero_genes = set()

    # 0 genes
    for person in people:
        if person not in one_gene and person not in two_genes:
            zero_genes.add(person)
    #print(zero_genes, one_gene, two_genes, have_trait)

    for person in zero_genes:
        # No given mother/father
        if people[person]['mother'] is None or people[person]['father'] is None:
            # Unconditional probability = PROBS["gene"]
            up = PROBS["gene"][0]

            if person in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][0][True]
            if person not in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][0][False]
            
            # Update p
            p[person] = up * cp

        # Given mother/father
        if people[person]['mother'] is not None or people[person]['father'] is not None:
            m = people[person]['mother']
            f = people[person]['father']

            if m in zero_genes:
                # Mutation probability = PROBS["mutation"]
                mump = 1 - PROBS["mutation"]
                mmp = PROBS["mutation"]
            
            if m in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                mump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                mmp = 0.5

            if m in two_genes:
                # Mutation probability = PROBS["mutation"]
                mump = PROBS["mutation"]
                mmp = 1 - PROBS["mutation"]

            if f in zero_genes:
                fump = 1 - PROBS["mutation"]
                fmp = PROBS["mutation"]
            
            if f in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                fump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                fmp = 0.5

            if f in two_genes:
                fump = PROBS["mutation"]
                fmp = 1 - PROBS["mutation"]
            #print("0", mump, mmp, fump, fmp)
            
            # 0 genes, but has trait (e.g. P(0 genes) * PROBS["trait"][0][True])
            if person in have_trait:
                p[person] = (mump * fump) * PROBS["trait"][0][True]
            # 0 genes, and has no trait (e.g. P(0 genes) * PROBS["trait"][0][False])
            if person not in have_trait:
                p[person] = (mump * fump) * PROBS["trait"][0][False]

    # 2 genes
    for person in two_genes:
        # No given mother/father
        if people[person]['mother'] is None or people[person]['father'] is None:
            # Unconditional probability = PROBS["gene"]
            up = PROBS["gene"][2]

            if person in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][2][True]
            if person not in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][2][False]
            
            # Update p
            p[person] = up * cp

        # Given mother/father
        if people[person]['mother'] is not None or people[person]['father'] is not None:
            m = people[person]['mother']
            f = people[person]['father']

            if m in zero_genes:
                # Mutation probability = PROBS["mutation"]
                mump = 1 - PROBS["mutation"]
                mmp = PROBS["mutation"]

            if m in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                mump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                mmp = 0.5
            
            if m in two_genes:
                # Mutation probability = PROBS["mutation"]
                mump = PROBS["mutation"]
                mmp = 1 - PROBS["mutation"]

            if f in zero_genes:
                fump = 1 - PROBS["mutation"]
                fmp = PROBS["mutation"]
            
            if f in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                fump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                fmp = 0.5

            if f in two_genes:
                fump = PROBS["mutation"]
                fmp = 1 - PROBS["mutation"]
            #print("2", mump, mmp, fump, fmp)
        
            # 2 genes, and has trait (e.g. P(2 genes) * PROBS["trait"][2][True])
            if person in have_trait:
                p[person] = (mmp * fmp) * PROBS["trait"][2][True]
            # 2 genes, but has no trait (e.g. P(2 genes) * PROBS["trait"][2][False])
            if person not in have_trait:
                p[person] = (mmp * fmp) * PROBS["trait"][2][False]

    # 1 gene
    for person in one_gene:
        # No given mother/father
        if people[person]['mother'] is None or people[person]['father'] is None:
            # Unconditional probability = PROBS["gene"]
            up = PROBS["gene"][1]

            if person in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][1][True]
            if person not in have_trait:
                # Conditional probability = PROBS["trait"]
                cp = PROBS["trait"][1][False]
            
            # Update p
            p[person] = up * cp

        # Given mother/father
        if people[person]['mother'] is not None or people[person]['father'] is not None:
            m = people[person]['mother']
            f = people[person]['father']

            if m in zero_genes:
                # Mutation probability = PROBS["mutation"]
                mump = 1 - PROBS["mutation"]
                mmp = PROBS["mutation"]

            if m in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                mump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                mmp = 0.5
                
            if m in two_genes:
                # Mutation probability = PROBS["mutation"]
                mump = PROBS["mutation"]
                mmp = 1 - PROBS["mutation"]

            if f in zero_genes:
                fump = 1 - PROBS["mutation"]
                fmp = PROBS["mutation"]
            
            if f in one_gene:
                # Passes down 0 = 0.49; passes down 1 but mutated = 0.01; 0.49+0.01 = 0.5
                fump = 0.5
                # Passes down 1 = 0.49; passes down 0 but mutated = 0.01; 0.49+0.01 = 0.5
                fmp = 0.5

            if f in two_genes:
                fump = PROBS["mutation"]
                fmp = 1 - PROBS["mutation"]
            #print("1", mump, mmp, fump, fmp)
        
            # 1 gene, and has trait (e.g. P(1 gene) * PROBS["trait"][1][True])
            if person in have_trait:
                p[person] = (mmp * fump + mump * fmp) * PROBS["trait"][1][True]
            # 1 gene, but has no trait (e.g. P(1 gene) * PROBS["trait"][1][False])
            if person not in have_trait:
                p[person] = (mmp * fump + mump * fmp) * PROBS["trait"][1][False]
                #print(person, (mmp * fump + mump * fmp), PROBS["trait"][1][False], (mmp * fump + mump * fmp) * PROBS["trait"][1][False])
    
    # Calculate joint probability
    jp = 1
    for person in p:
        jp *= p[person]

    return jp


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    zero_genes = set()

    # 0 genes
    for person in probabilities:
        if person not in one_gene and person not in two_genes:
            zero_genes.add(person)

    for person in probabilities:
        # 0 genes
        if person in zero_genes:
            probabilities[person]["gene"][0] += p
        # 1 gene
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        # 2 genes
        if person in two_genes:
            probabilities[person]["gene"][2] += p

        # Check given traits
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        if person not in have_trait:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    gene_total = 0
    trait_total = 0

    for person in probabilities:
        for n in range(3):
            p = probabilities[person]["gene"][n]
            gene_total += p
        
        # Normalize if total is not 1
        if gene_total != 1:
            for n in range(3):
                probabilities[person]["gene"][n] /= gene_total

        t = probabilities[person]["trait"][True]
        f = probabilities[person]["trait"][False]
        trait_total = t + f
        
        # Normalize if total is not 1
        if trait_total != 1:
            probabilities[person]["trait"][True] /= trait_total
            probabilities[person]["trait"][False] /= trait_total

        # Reset
        gene_total = 0
        trait_total = 0
            

if __name__ == "__main__":
    main()
