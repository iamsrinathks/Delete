kubectl get nodes \
  -o custom-columns=TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,ZONE:.metadata.labels.topology\.kubernetes\.io/zone \
  --no-headers | sort | uniq -c | sort -k2,2 -k3,3


  kubectl get nodes -o custom-columns=POOL:.metadata.labels.cloud\.google\.com/gke-nodepool,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,ZONE:.metadata.labels.topology\.kubernetes\.io/zone --no-headers \
    | sort | uniq -c

  kubectl get nodes -l node.kubernetes.io/instance-type=g2-standard-8 -L topology.kubernetes.io/zone --no-headers \
    | awk '{print $NF}' | sort | uniq -c

  gcloud compute instances list --project "$GKE_PROJECT" \
    --filter="labels.goog-k8s-cluster-name=$CLUSTER" \
    --format="value(machineType.basename(),zone.basename())" | sort | uniq -c


kubectl get nodes --no-headers \
  -o custom-columns=POOL:.metadata.labels.cloud\.google\.com/gke-nodepool,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,ZONE:.metadata.labels.topology\.kubernetes\.io/zone \
| awk '
{ key=$1" "$2; z=substr($3,length($3)); c[key,z]++; t[key]++ }
END {
  printf "%-30s %-24s %6s %5s %5s %5s\n","NODEPOOL","MACHINE_TYPE","TOTAL","A","B","C"
  for (k in t) { split(k,p," ");
    printf "%-30s %-24s %6d %5d %5d %5d\n", p[1], p[2], t[k], c[k,"a"], c[k,"b"], c[k,"c"] }
}' | (read h; echo "$h"; sort)
